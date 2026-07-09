"""
Fetch raw solar / geomagnetic observations for sunspot forecasting.

Sources (all daily, non-smoothed):
- Sunspot number: SILSO (1749-present)
- F10.7 solar flux: CelesTrak (1947-present)
- Kp / Ap geomagnetic indices: GFZ Potsdam (1932-present)

This is the only piece of the legacy pipeline retained in the package: it turns
remote data into the ``raw_multivariate_data.csv`` the trainer consumes.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

SILSO_DAILY_URL = "https://www.sidc.be/SILSO/INFO/sndtotcsv.php"
SILSO_MONTHLY_URL = "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv"
SUNSPOT_AREA_URL = "http://solarcyclescience.com/AR_Database/daily_area.txt"
F107_URL = "https://celestrak.org/SpaceData/SW-All.csv"
KP_AP_URL = "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"

REQUEST_TIMEOUT = 30


class SolarDataCollector:
    """Collects and merges solar / geomagnetic data into a single daily table."""

    def fetch_sunspot_data(self, start_year: int = 1749,
                           end_year: Optional[int] = None) -> pd.DataFrame:
        """Fetch daily (non-smoothed) sunspot numbers from SILSO.

        Returns a DataFrame with columns: date, sunspot_number, std_dev,
        num_observations, provisional. Returns an empty frame on network error.
        """
        end_year = end_year or datetime.now().year
        print(f"Fetching SILSO daily sunspot data {start_year}-{end_year}...")

        columns = ['date', 'sunspot_number', 'std_dev', 'num_observations', 'provisional']
        try:
            response = requests.get(SILSO_DAILY_URL, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"Error fetching SILSO data: {exc}")
            return pd.DataFrame(columns=columns)

        rows = []
        for line in response.text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(';')
            if len(parts) < 5:
                continue
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            if not (start_year <= year <= end_year):
                continue
            # SILSO encodes missing values as -1 (formatted as "-1", " -1.0", ...);
            # compare numerically, not as strings.
            sn = float(parts[4])
            sd = float(parts[5]) if len(parts) > 5 and parts[5].strip() else -1.0
            nobs = int(float(parts[6])) if len(parts) > 6 and parts[6].strip() else -1
            rows.append({
                'date': pd.Timestamp(year=year, month=month, day=day),
                'sunspot_number': sn if sn >= 0 else np.nan,
                'std_dev': sd if sd >= 0 else np.nan,
                'num_observations': nobs if nobs >= 0 else 0,
                'provisional': int(parts[7]) if len(parts) > 7 else 0,
            })

        df = pd.DataFrame(rows, columns=columns).sort_values('date').reset_index(drop=True)
        print(f"  fetched {len(df)} daily sunspot observations")
        return df

    def fetch_silso_monthly(self, start_year: int = 1749) -> pd.DataFrame:
        """Fetch the SILSO monthly-mean sunspot number (SN_m_tot_V2.0, 1749-present).

        This is the officially curated monthly series (the one used in Benson et
        al. 2020) - prefer it over hand-aggregating the daily file. Columns:
        date (month end), sunspot_number, std_dev, num_observations, provisional.
        Missing values (-1) are masked to NaN. Empty frame on network error.
        """
        print(f"Fetching SILSO monthly mean sunspot number from {start_year}...")
        columns = ['date', 'sunspot_number', 'std_dev', 'num_observations', 'provisional']
        try:
            response = requests.get(SILSO_MONTHLY_URL, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"Error fetching SILSO monthly data: {exc}")
            return pd.DataFrame(columns=columns)

        rows = []
        # Format: year;month;decimal_date;ssn;std_dev;n_obs;provisional
        for line in response.text.strip().splitlines():
            parts = [p.strip() for p in line.split(';')]
            if len(parts) < 4 or not parts[0].isdigit():
                continue
            year, month = int(parts[0]), int(parts[1])
            if year < start_year:
                continue
            sn = float(parts[3])
            sd = float(parts[4]) if len(parts) > 4 and parts[4] else -1.0
            nobs = int(float(parts[5])) if len(parts) > 5 and parts[5] else -1
            rows.append({
                'date': pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0),
                'sunspot_number': sn if sn >= 0 else np.nan,
                'std_dev': sd if sd >= 0 else np.nan,
                'num_observations': nobs if nobs >= 0 else 0,
                'provisional': int(parts[6]) if len(parts) > 6 and parts[6] else 0,
            })

        df = pd.DataFrame(rows, columns=columns).sort_values('date').reset_index(drop=True)
        print(f"  fetched {len(df)} monthly sunspot observations")
        return df

    def fetch_sunspot_area(self, start_year: int = 1874) -> pd.DataFrame:
        """Fetch daily total sunspot area (Hathaway/Upton, 1874-present).

        Whitespace-separated columns: YYYY MM DD Total North South, with -1.0
        sentinels for missing days (masked to NaN). Returns date + sunspot_area.
        Empty frame on network error.
        """
        print(f"Fetching daily total sunspot area from {start_year}...")
        columns = ['date', 'sunspot_area']
        try:
            response = requests.get(SUNSPOT_AREA_URL, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"Error fetching sunspot area data: {exc}")
            return pd.DataFrame(columns=columns)

        rows = []
        for line in response.text.strip().splitlines():
            parts = line.split()
            if len(parts) < 4 or not parts[0].isdigit():
                continue
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            if year < start_year:
                continue
            try:
                area = float(parts[3])
            except ValueError:
                continue
            rows.append({
                'date': pd.Timestamp(year=year, month=month, day=day),
                'sunspot_area': area if area >= 0 else np.nan,
            })

        df = pd.DataFrame(rows, columns=columns).sort_values('date').reset_index(drop=True)
        print(f"  fetched {len(df)} daily sunspot area observations")
        return df

    def fetch_f107_data(self, start_year: int = 1947) -> pd.DataFrame:
        """Fetch F10.7 solar flux from CelesTrak. Empty frame on error."""
        print(f"Fetching F10.7 solar flux from {start_year}...")
        try:
            response = requests.get(F107_URL, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"Error fetching F10.7 data: {exc}")
            return pd.DataFrame(columns=['date'])

        df = pd.read_csv(io.StringIO(response.text))
        if 'DATE' not in df.columns:
            print("  F10.7 response missing DATE column")
            return pd.DataFrame(columns=['date'])

        df['date'] = pd.to_datetime(df['DATE'])
        df = df[df['date'].dt.year >= start_year]
        flux_cols = ['date'] + [c for c in df.columns
                                if 'F10.7' in c.upper() or 'FLUX' in c.upper()]
        df = df[flux_cols].sort_values('date').reset_index(drop=True)
        print(f"  fetched {len(df)} F10.7 observations")
        return df

    def fetch_kp_ap_data(self, start_year: int = 1932) -> pd.DataFrame:
        """Fetch Kp / Ap geomagnetic indices from GFZ Potsdam. Empty frame on error."""
        print(f"Fetching Kp/Ap geomagnetic data from {start_year}...")
        try:
            response = requests.get(KP_AP_URL, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"Error fetching Kp/Ap data: {exc}")
            return pd.DataFrame(columns=['date', 'kp_sum', 'ap_avg'])

        rows = []
        for line in response.text.strip().splitlines():
            if not line.strip() or line.startswith('#') or len(line) <= 50:
                continue
            try:
                year, month, day = int(line[0:4]), int(line[4:6]), int(line[6:8])
                if year < start_year:
                    continue
                rows.append({
                    'date': pd.Timestamp(year=year, month=month, day=day),
                    'kp_sum': float(line[17:20]) if line[17:20].strip() else np.nan,
                    'ap_avg': float(line[21:24]) if line[21:24].strip() else np.nan,
                })
            except (ValueError, IndexError):
                continue

        df = pd.DataFrame(rows, columns=['date', 'kp_sum', 'ap_avg'])
        df = df.sort_values('date').reset_index(drop=True)
        print(f"  fetched {len(df)} Kp/Ap observations")
        return df

    def build_dataset(self, start_year: int = 1749,
                      end_year: Optional[int] = None) -> pd.DataFrame:
        """Merge all sources into one date-aligned daily table.

        Each source starts at its earliest available year; F10.7 and Kp/Ap are
        forward-filled to cover the daily sunspot index.
        """
        sunspot = self.fetch_sunspot_data(start_year, end_year)
        f107 = self.fetch_f107_data(1947)
        kp_ap = self.fetch_kp_ap_data(1932)

        combined = sunspot.copy()
        if not f107.empty:
            f107 = f107.rename(columns=lambda c: f'f107_{c}' if c != 'date' else c)
            combined = combined.merge(f107, on='date', how='left')
        if not kp_ap.empty:
            combined = combined.merge(kp_ap, on='date', how='left')

        combined = combined.sort_values('date').reset_index(drop=True)

        # Forward-fill the lower-cadence exogenous series.
        fill_cols = [c for c in combined.columns
                     if c.startswith('f107') or any(k in c.lower() for k in ('kp', 'ap'))]
        if fill_cols:
            combined[fill_cols] = combined[fill_cols].ffill()

        print(f"Built dataset: {len(combined)} rows, {len(combined.columns)} columns, "
              f"{combined['date'].min().date()} to {combined['date'].max().date()}")
        return combined


def main() -> None:
    """CLI: fetch all datasets.

    Writes:
    - data/silso_monthly.csv           (curated monthly SSN, 1749-present - primary)
    - data/sunspot_area_monthly.csv    (monthly total sunspot area, 1874-present)
    - data/raw_multivariate_data.csv   (daily SSN + F10.7 + Kp/Ap, for precursor runs)
    """
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    collector = SolarDataCollector()

    monthly = collector.fetch_silso_monthly(start_year=1749)
    if not monthly.empty:
        monthly.to_csv(data_dir / "silso_monthly.csv", index=False)
        print(f"Saved SILSO monthly series to {data_dir / 'silso_monthly.csv'}")

    area = collector.fetch_sunspot_area(start_year=1874)
    if not area.empty:
        # Aggregate to monthly means with sentinels already masked to NaN.
        area_monthly = (area.set_index('date')['sunspot_area']
                        .resample('ME').mean().reset_index())
        area_monthly.to_csv(data_dir / "sunspot_area_monthly.csv", index=False)
        print(f"Saved monthly sunspot area to {data_dir / 'sunspot_area_monthly.csv'}")

    dataset = collector.build_dataset(start_year=1749)
    dataset.to_csv(data_dir / "raw_multivariate_data.csv", index=False)
    print(f"Saved daily multivariate dataset to {data_dir / 'raw_multivariate_data.csv'}")


if __name__ == "__main__":
    main()
