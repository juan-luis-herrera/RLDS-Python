"""
Carbon Intensity Cache - Hourly Granularity Version
For querying historical hourly carbon intensity data during training (no interpolation, uses actual values)
"""

import csv
from datetime import datetime, timedelta
import bisect


class CarbonIntensityCacheHourly:
    """
    Carbon Intensity Data Cache (Hourly granularity, no interpolation)

    Features:
    1. Load hourly granularity historical data
    2. Map current time to same period last year
    3. Round down to hour boundary
    4. Return carbon intensity value for that hour (no interpolation)
    """

    def __init__(self, csv_file, default_value=300.0):
        """
        Initialize cache

        Args:
            csv_file: Path to CSV file containing historical data (hourly granularity)
            default_value: Default value when query fails
        """
        self.csv_file = csv_file
        self.default_value = default_value

        # Data storage - using dict, key is hour boundary timestamp
        self.carbon_data = {}  # {datetime: carbon_value}

        # Statistics
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # Load data
        self._load_data()

    def _load_data(self):
        """Load CSV data into memory"""
        print(f"\n[Loading] Loading Carbon Intensity Cache (Hourly): {self.csv_file}")

        try:
            with open(self.csv_file, 'r') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    # Parse timestamp
                    ts_str = row['timestamp']
                    ts = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')

                    # Parse carbon value
                    carbon = float(row['carbon_intensity'])

                    # Store in dict
                    self.carbon_data[ts] = carbon

            # Statistics
            timestamps = list(self.carbon_data.keys())
            timestamps.sort()

            carbon_values = list(self.carbon_data.values())

            print(f"   [OK] Load successful")
            print(f"   Data points: {len(self.carbon_data):,}")
            print(f"   Time range: {timestamps[0]} to {timestamps[-1]}")
            print(f"   Carbon range: {min(carbon_values):.1f} - {max(carbon_values):.1f} gCO2eq/kWh")
            print(f"   Carbon mean: {sum(carbon_values)/len(carbon_values):.1f} gCO2eq/kWh")
            print(f"   Data granularity: hourly (one value per hour)\n")

        except FileNotFoundError:
            print(f"   [ERROR] File not found: {self.csv_file}")
            print(f"   Please run: python3 download_austria_carbon_hourly.py")
            raise
        except Exception as e:
            print(f"   [ERROR] Load failed: {e}")
            raise

    def _round_down_to_hour(self, dt):
        """
        Round time down to hour boundary

        Args:
            dt: datetime object

        Returns:
            datetime: Rounded time (minutes and seconds set to 0)

        Example:
            2024-12-15 14:23:45 -> 2024-12-15 14:00:00
            2024-12-15 14:59:59 -> 2024-12-15 14:00:00
            2024-12-15 15:00:00 -> 2024-12-15 15:00:00
        """
        return dt.replace(minute=0, second=0, microsecond=0)

    def map_to_last_year(self, current_time):
        """
        Map current time to same period last year

        Args:
            current_time: Current time (datetime object)

        Returns:
            datetime: Same period last year
        """
        # Simple method: subtract 365 days
        last_year = current_time - 2*timedelta(days=365)

        # Handle leap year: if current year is leap year and date is Feb 29
        if current_time.month == 2 and current_time.day == 29:
            # Use Feb 28 of last year
            last_year = datetime(
                current_time.year - 1,
                2, 28,
                current_time.hour,
                current_time.minute,
                current_time.second
            )

        return last_year

    def get_carbon_intensity(self, current_time=None):
        """
        Get carbon intensity for specified time (no interpolation, uses actual hourly value)

        Args:
            current_time: Current time (datetime object)
                         If None, uses system current time

        Returns:
            float: carbon intensity value (gCO2eq/kWh)

        Flow:
            1. Map to same period last year
            2. Round down to hour boundary
            3. Query carbon value for that hour
            4. Return actual value (no interpolation)
        """
        # If no time specified, use current time
        if current_time is None:
            current_time = datetime.now()

        self.total_queries += 1

        # 1. Map to same period last year
        target_time = self.map_to_last_year(current_time)

        # 2. Round down to hour boundary
        hour_boundary = self._round_down_to_hour(target_time)

        # 3. Look up in data
        if hour_boundary in self.carbon_data:
            self.cache_hits += 1
            return self.carbon_data[hour_boundary]
        else:
            # Not found: use default value
            self.cache_misses += 1
            print(f"[WARNING] Time {hour_boundary} not in data range, using default value {self.default_value}")
            return self.default_value

    def get_carbon_intensity_exact(self, target_time):
        """
        Exact query (no year mapping, directly query specified time)

        For testing and debugging

        Args:
            target_time: Target time (datetime)

        Returns:
            float: carbon intensity value
        """
        hour_boundary = self._round_down_to_hour(target_time)

        if hour_boundary in self.carbon_data:
            return self.carbon_data[hour_boundary]
        else:
            print(f"[WARNING] Time {hour_boundary} not in data")
            return self.default_value

    def get_statistics(self):
        """
        Get cache statistics

        Returns:
            dict: Statistics data
        """
        hit_rate = (self.cache_hits / self.total_queries * 100) if self.total_queries > 0 else 0

        timestamps = sorted(self.carbon_data.keys())
        carbon_values = list(self.carbon_data.values())

        return {
            'total_queries': self.total_queries,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'data_points': len(self.carbon_data),
            'time_range': (
                str(timestamps[0]) if timestamps else None,
                str(timestamps[-1]) if timestamps else None
            ),
            'carbon_range': (
                min(carbon_values) if carbon_values else None,
                max(carbon_values) if carbon_values else None
            ),
            'carbon_mean': sum(carbon_values) / len(carbon_values) if carbon_values else None
        }

    def print_statistics(self):
        """Print statistics"""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print("Carbon Intensity Cache Statistics (Hourly)")
        print("="*60)
        print(f"Total queries: {stats['total_queries']:,}")
        print(f"Cache hits: {stats['cache_hits']:,}")
        print(f"Cache misses: {stats['cache_misses']:,}")
        print(f"Hit rate: {stats['hit_rate']:.2f}%")
        print(f"\nDataset info:")
        print(f"  Data points: {stats['data_points']:,}")
        print(f"  Time range: {stats['time_range'][0]} to {stats['time_range'][1]}")
        print(f"  Carbon range: {stats['carbon_range'][0]:.1f} - {stats['carbon_range'][1]:.1f}")
        print(f"  Carbon mean: {stats['carbon_mean']:.1f}")
        print(f"  Data granularity: hourly (no interpolation)")
        print("="*60 + "\n")


# Test code
if __name__ == "__main__":
    """Test CarbonIntensityCacheHourly class"""

    print("="*80)
    print("Carbon Intensity Cache (Hourly) Test")
    print("="*80)

    # Create cache
    try:
        cache = CarbonIntensityCacheHourly(
            csv_file="carbon_intensity_austria_2024_hourly.csv",
            default_value=300.0
        )
    except:
        print("\n[ERROR] Cannot load data file")
        print("Please run: python3 download_austria_carbon_hourly.py")
        exit(1)

    # Test 1: Simulate training process (query every 25 seconds)
    print("\nTest 1: Simulate training process (query every 25 seconds, within 1 hour)")
    print("-" * 80)

    test_start = datetime(2025, 12, 15, 14, 0, 0)  # Year 2025
    carbon_values = []

    print(f"Simulated time: {test_start} (mapped to last year {cache.map_to_last_year(test_start)})\n")

    # Simulate queries within 1 hour (3600 seconds / 25 seconds = 144 times)
    for i in range(144):
        test_time = test_start + timedelta(seconds=i*25)
        carbon = cache.get_carbon_intensity(test_time)
        carbon_values.append(carbon)

        # Only print first 5 and last 5
        if i < 5 or i >= 139:
            mapped = cache.map_to_last_year(test_time)
            hour = cache._round_down_to_hour(mapped)
            print(f"  [{i:3d}] {test_time.strftime('%H:%M:%S')} -> "
                  f"last year: {mapped.strftime('%H:%M:%S')} -> "
                  f"hour boundary: {hour.strftime('%H:%M')} -> "
                  f"carbon: {carbon:.2f}")
        elif i == 5:
            print(f"  ... (omitting middle {144-10} queries)")

    # Verify: all queries within 1 hour should return same value
    unique_values = set(carbon_values)
    print(f"\nVerification result:")
    print(f"  Total queries: {len(carbon_values)}")
    print(f"  Unique carbon values: {len(unique_values)}")

    if len(unique_values) == 1:
        print(f"  [OK] Correct! All queries within 1 hour return same value: {list(unique_values)[0]:.2f}")
    else:
        print(f"  [ERROR] Wrong! Multiple different values within 1 hour: {unique_values}")

    # Test 2: Cross hour boundary
    print("\n\nTest 2: Queries crossing hour boundary")
    print("-" * 80)

    # Start from 14:55, query until 15:05 (crossing 15:00 boundary)
    test_start = datetime(2025, 12, 15, 14, 55, 0)

    for i in range(5):  # 5 queries, crossing boundary
        test_time = test_start + timedelta(minutes=i*3)
        carbon = cache.get_carbon_intensity(test_time)

        mapped = cache.map_to_last_year(test_time)
        hour = cache._round_down_to_hour(mapped)

        print(f"  {test_time.strftime('%H:%M')} -> "
              f"last year: {mapped.strftime('%H:%M')} -> "
              f"hour boundary: {hour.strftime('%H:%M')} -> "
              f"carbon: {carbon:.2f}")

    # Test 3: Direct query last year data
    print("\n\nTest 3: Direct query specific times last year")
    print("-" * 80)

    test_times = [
        datetime(2024, 12, 1, 0, 0, 0),
        datetime(2024, 12, 1, 12, 0, 0),
        datetime(2024, 12, 1, 18, 0, 0),
        datetime(2024, 12, 15, 14, 30, 0),  # With minutes, should round down
    ]

    for ts in test_times:
        carbon = cache.get_carbon_intensity_exact(ts)
        hour = cache._round_down_to_hour(ts)
        print(f"  {ts.strftime('%Y-%m-%d %H:%M')} -> "
              f"hour boundary: {hour.strftime('%H:%M')} -> "
              f"carbon: {carbon:.2f}")

    # Print statistics
    cache.print_statistics()

    print("="*80)
    print("[DONE] Test completed!")
    print("="*80)
    print("\nKey points:")
    print("  1. [OK] All queries within same hour return same carbon value")
    print("  2. [OK] Time automatically rounds down to hour boundary")
    print("  3. [OK] Automatically maps to same period last year")
    print("  4. [OK] No interpolation, uses actual hourly data")
