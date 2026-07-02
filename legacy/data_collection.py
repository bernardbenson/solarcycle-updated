"""
Data collection module for sunspot prediction with actual (non-smoothed) data.
Fetches latest SILSO sunspot numbers, F10.7 solar flux, and geomagnetic indices.
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Tuple, Optional
import io


class SolarDataCollector:
    """Collects solar and geomagnetic data for multivariate sunspot prediction."""
    
    def __init__(self):
        self.silso_base_url = "https://www.sidc.be/SILSO/INFO/sndtotcsv.php"
        self.silso_daily_url = "https://www.sidc.be/SILSO/INFO/sndtotcsv.php"
        self.f107_url = "https://celestrak.org/SpaceData/SW-All.csv"
        self.kp_ap_url = "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
        
    def fetch_silso_sunspot_data(self, start_year: int = 1749, end_year: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch actual (non-smoothed) daily sunspot numbers from SILSO.
        
        Args:
            start_year: Starting year for data collection
            end_year: Ending year (None for current year)
            
        Returns:
            DataFrame with columns: date, sunspot_number, std_dev, num_observations, provisional
        """
        if end_year is None:
            end_year = datetime.now().year
            
        print(f"Fetching SILSO daily sunspot data from {start_year} to {end_year}...")
        
        try:
            # SILSO daily sunspot number data format:
            # Year Month Day Date_fraction Sunspot_number Std_dev Num_observations Provisional
            url = "https://www.sidc.be/SILSO/INFO/sndtotcsv.php"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse the CSV data
            data = []
            for line in response.text.strip().split('\n'):
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split(';')
                    if len(parts) >= 5:
                        year = int(parts[0])
                        month = int(parts[1]) 
                        day = int(parts[2])
                        sunspot_num = float(parts[4]) if parts[4] != '-1' else np.nan
                        std_dev = float(parts[5]) if len(parts) > 5 and parts[5] != '-1' else np.nan
                        num_obs = int(parts[6]) if len(parts) > 6 and parts[6] != '-1' else 0
                        provisional = int(parts[7]) if len(parts) > 7 else 0
                        
                        if start_year <= year <= end_year:
                            date = pd.to_datetime(f"{year}-{month:02d}-{day:02d}")
                            data.append({
                                'date': date,
                                'sunspot_number': sunspot_num,
                                'std_dev': std_dev,
                                'num_observations': num_obs,
                                'provisional': provisional
                            })
            
            df = pd.DataFrame(data)
            df = df.sort_values('date').reset_index(drop=True)
            print(f"Successfully fetched {len(df)} daily sunspot observations")
            return df
            
        except Exception as e:
            print(f"Error fetching SILSO data: {e}")
            # Fallback to alternative URL or local backup
            return self._fetch_silso_backup()
    
    def _fetch_silso_backup(self) -> pd.DataFrame:
        """Backup method to fetch SILSO data from alternative source."""
        try:
            # Alternative: Fetch from LASP LISIRD
            url = "https://lasp.colorado.edu/lisird/data/international_sunspot_number/"
            print("Trying alternative SILSO data source...")
            
            # This would need specific implementation based on LISIRD API
            # For now, return empty dataframe with proper structure
            return pd.DataFrame(columns=['date', 'sunspot_number', 'std_dev', 'num_observations', 'provisional'])
            
        except Exception as e:
            print(f"Backup data source also failed: {e}")
            return pd.DataFrame(columns=['date', 'sunspot_number', 'std_dev', 'num_observations', 'provisional'])
    
    def fetch_f107_data(self, start_year: int = 1947) -> pd.DataFrame:
        """
        Fetch F10.7 solar flux data from CelesTrak.
        
        Args:
            start_year: Starting year for data collection
            
        Returns:
            DataFrame with F10.7 flux values
        """
        print(f"Fetching F10.7 solar flux data from {start_year}...")
        
        try:
            response = requests.get(self.f107_url, timeout=30)
            response.raise_for_status()
            
            # Parse CSV data
            df = pd.read_csv(io.StringIO(response.text))
            
            # Convert date columns and filter by year
            if 'DATE' in df.columns:
                df['date'] = pd.to_datetime(df['DATE'])
                df = df[df['date'].dt.year >= start_year]
                
                # Select relevant F10.7 columns
                f107_cols = ['date']
                for col in df.columns:
                    if 'F10.7' in col.upper() or 'FLUX' in col.upper():
                        f107_cols.append(col)
                
                df = df[f107_cols].copy()
                df = df.sort_values('date').reset_index(drop=True)
                
                print(f"Successfully fetched {len(df)} F10.7 flux observations")
                return df
                
        except Exception as e:
            print(f"Error fetching F10.7 data: {e}")
            return pd.DataFrame(columns=['date', 'f107_flux'])
    
    def fetch_kp_ap_data(self, start_year: int = 1932) -> pd.DataFrame:
        """
        Fetch Kp and Ap geomagnetic indices from GFZ Potsdam.
        
        Args:
            start_year: Starting year for data collection
            
        Returns:
            DataFrame with Kp and Ap indices
        """
        print(f"Fetching Kp/Ap geomagnetic data from {start_year}...")
        
        try:
            response = requests.get(self.kp_ap_url, timeout=30)
            response.raise_for_status()
            
            # Parse the fixed-width format file
            data = []
            for line in response.text.strip().split('\n'):
                if line.strip() and not line.startswith('#') and len(line) > 50:
                    try:
                        year = int(line[0:4])
                        month = int(line[4:6])
                        day = int(line[6:8])
                        
                        if year >= start_year:
                            date = pd.to_datetime(f"{year}-{month:02d}-{day:02d}")
                            
                            # Extract Kp and Ap values (positions may vary)
                            kp_sum = float(line[17:20]) if line[17:20].strip() else np.nan
                            ap_avg = float(line[21:24]) if line[21:24].strip() else np.nan
                            
                            data.append({
                                'date': date,
                                'kp_sum': kp_sum,
                                'ap_avg': ap_avg
                            })
                    except (ValueError, IndexError):
                        continue
            
            df = pd.DataFrame(data)
            df = df.sort_values('date').reset_index(drop=True)
            print(f"Successfully fetched {len(df)} Kp/Ap observations")
            return df
            
        except Exception as e:
            print(f"Error fetching Kp/Ap data: {e}")
            return pd.DataFrame(columns=['date', 'kp_sum', 'ap_avg'])
    
    def create_multivariate_dataset(self, start_year: int = 1749, end_year: Optional[int] = None) -> pd.DataFrame:
        """
        Create a combined multivariate dataset with all solar and geomagnetic indices.
        Uses maximum available historical data for each source:
        - Sunspot numbers: 1749-present
        - F10.7 flux: 1947-present  
        - Kp/Ap indices: 1932-present
        
        Args:
            start_year: Starting year (default 1749 for maximum sunspot history)
            end_year: Ending year (None for current year)
            
        Returns:
            Combined DataFrame with all features aligned by date
        """
        print("Creating comprehensive multivariate dataset with maximum historical data...")
        
        # Fetch all data sources with their natural starting points
        print(f"Fetching sunspot data from {start_year}...")
        sunspot_df = self.fetch_silso_sunspot_data(start_year, end_year)
        
        print("Fetching F10.7 data from 1947 (earliest available)...")
        f107_df = self.fetch_f107_data(1947)  # F10.7 starts in 1947
        
        print("Fetching Kp/Ap data from 1932 (earliest available)...")
        kp_ap_df = self.fetch_kp_ap_data(1932)  # Kp/Ap starts in 1932
        
        # Start with sunspot data as base
        combined_df = sunspot_df.copy()
        
        # Merge F10.7 data
        if not f107_df.empty:
            f107_df = f107_df.rename(columns=lambda x: f'f107_{x}' if x != 'date' else x)
            combined_df = combined_df.merge(f107_df, on='date', how='left')
        
        # Merge Kp/Ap data
        if not kp_ap_df.empty:
            combined_df = combined_df.merge(kp_ap_df, on='date', how='left')
        
        # Sort by date and fill forward missing values
        combined_df = combined_df.sort_values('date').reset_index(drop=True)
        
        # Forward fill geomagnetic indices (they're often reported daily)
        geomag_cols = [col for col in combined_df.columns if any(x in col.lower() for x in ['kp', 'ap'])]
        if geomag_cols:
            combined_df[geomag_cols] = combined_df[geomag_cols].ffill()
        
        # Forward fill F10.7 values
        f107_cols = [col for col in combined_df.columns if 'f107' in col.lower()]
        if f107_cols:
            combined_df[f107_cols] = combined_df[f107_cols].ffill()
        
        print(f"Created multivariate dataset with {len(combined_df)} observations and {len(combined_df.columns)} features")
        print(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        
        # Show data coverage for each source
        print("\n=== DATA COVERAGE BY SOURCE ===")
        print(f"Sunspot numbers: {len(sunspot_df)} observations ({sunspot_df['date'].min().strftime('%Y-%m-%d')} to {sunspot_df['date'].max().strftime('%Y-%m-%d')})")
        if not f107_df.empty:
            print(f"F10.7 solar flux: {len(f107_df)} observations ({f107_df['date'].min().strftime('%Y-%m-%d')} to {f107_df['date'].max().strftime('%Y-%m-%d')})")
        if not kp_ap_df.empty:
            print(f"Kp/Ap indices: {len(kp_ap_df)} observations ({kp_ap_df['date'].min().strftime('%Y-%m-%d')} to {kp_ap_df['date'].max().strftime('%Y-%m-%d')})")
        
        # Calculate data completeness
        total_days = (combined_df['date'].max() - combined_df['date'].min()).days + 1
        completeness = len(combined_df) / total_days * 100
        print(f"\nData completeness: {completeness:.1f}% ({len(combined_df)} out of {total_days} possible days)")
        
        print(f"\nFeature categories: {len([c for c in combined_df.columns if c.startswith('f107')])} F10.7 features, {len([c for c in combined_df.columns if any(x in c.lower() for x in ['kp', 'ap'])])} geomagnetic features")
        
        return combined_df


def main():
    """Example usage of the data collector."""
    collector = SolarDataCollector()
    
    # Create multivariate dataset from 1749 (full sunspot history) to present
    dataset = collector.create_multivariate_dataset(start_year=1749)
    
    # Display basic statistics
    print("\nDataset Info:")
    print(dataset.info())
    print("\nFirst few rows:")
    print(dataset.head())
    print("\nBasic statistics:")
    print(dataset.describe())
    
    # Save to CSV
    dataset.to_csv('multivariate_solar_data.csv', index=False)
    print("\nDataset saved to 'multivariate_solar_data.csv'")


if __name__ == "__main__":
    main()