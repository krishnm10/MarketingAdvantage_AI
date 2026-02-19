#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════
🧪 PLATFORM-INDEPENDENT DEDUPLICATION TEST SUITE
Works on Windows, Linux, macOS
═══════════════════════════════════════════════════════════
"""

import os
import sys
import time
import requests
from pathlib import Path
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000/api/v2/api/v2/ingestion/upload"
TEST_DIR = Path("./test_files")
SLEEP_TIME = 2

# Colors for terminal output (works on Windows 10+ and Unix)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def enable_windows_colors():
        """Enable ANSI colors on Windows 10+"""
        if sys.platform == 'win32':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

Colors.enable_windows_colors()

def print_header(text):
    print(f"\n{Colors.BLUE}{'═' * 70}{Colors.ENDC}")
    print(f"{Colors.BLUE}{text}{Colors.ENDC}")
    print(f"{Colors.BLUE}{'═' * 70}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.YELLOW}{text}{Colors.ENDC}")

def create_test_files():
    """Create all test files"""
    print_header("📁 Creating Test Files")

    # Create test directory
    TEST_DIR.mkdir(exist_ok=True)

    test_files = {
        # ═══════════════════════════════════════════════════════════
        # TEST SET 1: LAYER 1 TESTING (Normalized Hash)
        # ═══════════════════════════════════════════════════════════
        "test_layer1_original.txt": """MarketingAdvantage AI is a powerful business intelligence platform.
We help companies analyze their marketing data efficiently.
Our platform integrates with multiple data sources seamlessly.
Revenue analytics and customer insights are our core features.""",

        "test_layer1_case_variation.txt": """MARKETINGADVANTAGE AI IS A POWERFUL BUSINESS INTELLIGENCE PLATFORM.
WE HELP COMPANIES ANALYZE THEIR MARKETING DATA EFFICIENTLY.
OUR PLATFORM INTEGRATES WITH MULTIPLE DATA SOURCES SEAMLESSLY.
REVENUE ANALYTICS AND CUSTOMER INSIGHTS ARE OUR CORE FEATURES.""",

        "test_layer1_space_variation.txt": """MarketingAdvantage   AI   is   a   powerful   business   intelligence   platform.
We    help    companies    analyze    their    marketing    data    efficiently.
Our  platform  integrates  with  multiple  data  sources  seamlessly.
Revenue     analytics     and     customer     insights     are     our     core     features.""",

        "test_layer1_punctuation_variation.txt": """MarketingAdvantage AI is a powerful business intelligence platform!
We help companies analyze their marketing data efficiently?
Our platform integrates with multiple data sources seamlessly...
Revenue analytics and customer insights are our core features!!!""",

        # ═══════════════════════════════════════════════════════════
        # TEST SET 2: LAYER 2 TESTING (Semantic Similarity)
        # ═══════════════════════════════════════════════════════════
        "test_layer2_original.txt": """Our company's revenue increased by 25 percent in the fourth quarter of 2024.
The marketing campaign generated significant customer engagement last month.
We plan to launch three new products in the upcoming fiscal year.""",

        "test_layer2_paraphrase.txt": """In Q4 2024, our company saw a 25% revenue growth.
Last month's marketing initiative drove substantial customer interaction.
Three product launches are scheduled for the next financial year.""",

        "test_layer2_similar.txt": """Revenue went up by one quarter in the last quarter of 2024.
Our marketing efforts resulted in high customer participation recently.
We're introducing three new offerings in the coming year.""",

        # ═══════════════════════════════════════════════════════════
        # TEST SET 3: LAYER 3 TESTING (Cross-File Deduplication)
        # ═══════════════════════════════════════════════════════════
        "test_layer3_report_q1.txt": """Company Overview: Founded in 2020, MarketingAdvantage AI serves over 10,000 customers worldwide.
Mission Statement: To revolutionize business intelligence through advanced AI-powered analytics.
Core Values: Innovation, Customer Success, Data Privacy, Continuous Improvement.""",

        "test_layer3_report_q2.txt": """Company Overview: Founded in 2020, MarketingAdvantage AI serves over 10,000 customers worldwide.
Mission Statement: To revolutionize business intelligence through advanced AI-powered analytics.
Q2 2024 Highlights: 15% revenue growth, 2000 new customers, 3 product launches.""",

        "test_layer3_report_q3.txt": """Company Overview: Founded in 2020, MarketingAdvantage AI serves over 10,000 customers worldwide.
Q3 2024 Performance: Revenue $50M, Customer Satisfaction 95%, Market Share 12%.
Core Values: Innovation, Customer Success, Data Privacy, Continuous Improvement.""",

        # ═══════════════════════════════════════════════════════════
        # TEST SET 4: CSV FILE TESTING
        # ═══════════════════════════════════════════════════════════
        "test_customers_jan.csv": """customer_id,name,email,revenue
1001,Acme Corp,acme@example.com,50000
1002,TechStart Inc,techstart@example.com,75000
1003,Global Solutions,global@example.com,100000""",

        "test_customers_feb.csv": """customer_id,name,email,revenue
1001,Acme Corp,acme@example.com,50000
1004,Innovation Labs,innovation@example.com,60000
1005,Digital Dynamics,digital@example.com,80000""",

        # ═══════════════════════════════════════════════════════════
        # TEST SET 5: MIXED CONTENT TESTING
        # ═══════════════════════════════════════════════════════════
        "test_mixed_content_a.txt": """Executive Summary: Q4 2024 Performance Review

Key Metrics:
- Total Revenue: $10 million USD
- Customer Acquisition: 5000 new customers
- Market Penetration: 15% growth in target segments

Strategic Initiatives:
We are expanding our AI capabilities with new machine learning models.
The product development team is focused on enhancing user experience.
Customer success metrics show 92% satisfaction rate across all segments.""",

        "test_mixed_content_b.txt": """Executive Summary: Q4 2024 Performance Review

Financial Highlights:
- Annual Recurring Revenue: $8 million USD
- Customer Retention Rate: 94% year-over-year
- Operating Margin: 22% improvement

Strategic Initiatives:
We are expanding our AI capabilities with new machine learning models.
The product development team is focused on enhancing user experience.
New partnerships announced with three Fortune 500 companies this quarter.""",

        # ═══════════════════════════════════════════════════════════
        # TEST SET 6: EDGE CASES
        # ═══════════════════════════════════════════════════════════
        "test_edge_empty.txt": """

""",

        "test_edge_short.txt": """Hi
OK
Yes""",
    }

    # Create repeated content file separately
    repeated_content = "This is a test sentence for deduplication testing purposes.\n" * 100
    test_files["test_edge_long_repeated.txt"] = repeated_content

    # Write all test files
    created_count = 0
    for filename, content in test_files.items():
        file_path = TEST_DIR / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        created_count += 1

    print_success(f"Created {created_count} test files in {TEST_DIR}")
    return list(test_files.keys())

def check_api_status():
    """Check if API is running"""
    print_info("🔍 Checking if API is running...")
    try:
        response = requests.get(API_URL.replace('/ingest', '/health'), timeout=5)
        print_success("API is running")
        return True
    except requests.exceptions.RequestException:
        # Try a simple connection test
        try:
            response = requests.post(API_URL, timeout=5)
            # Even if it returns error, API is at least responding
            print_success("API is running")
            return True
        except requests.exceptions.RequestException:
            print_error("API not responding at " + API_URL)
            print_info("\nPlease start your ingestion service first:")
            print_info("  python -m app.main")
            return False

def upload_file(file_path, test_name, expected):
    """Upload a file to the API"""
    print(f"\n{Colors.BLUE}{'━' * 70}{Colors.ENDC}")
    print(f"{Colors.YELLOW}📤 Uploading: {test_name}{Colors.ENDC}")
    print(f"   File: {file_path.name}")
    print(f"   Expected: {expected}")
    print(f"{Colors.BLUE}{'━' * 70}{Colors.ENDC}")

    try:
        with open(file_path, 'rb') as f:
            files = {'file': (file_path.name, f)}
            response = requests.post(API_URL, files=files, timeout=30)

        if response.status_code in [200, 201]:
            print_success("Upload successful")
            return True
        else:
            print_error(f"Upload failed: HTTP {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
    except Exception as e:
        print_error(f"Upload error: {str(e)}")
        return False

def run_tests():
    """Main test execution"""
    print_header("🧪 MarketingAdvantage AI - Deduplication Testing Suite")

    # Create test files
    test_files = create_test_files()

    # Check API status
    if not check_api_status():
        return False

    print_info(f"\n🚀 Starting deduplication tests...\n")
    time.sleep(1)

    # Define test cases with expected results
    test_cases = [
        # Test Set 1: Layer 1 Testing
        {
            'name': 'TEST SET 1: Layer 1 (Normalized Hash Deduplication)',
            'files': [
                ('test_layer1_original.txt', 'Layer 1 Original', '4 chunks stored, 0 duplicates'),
                ('test_layer1_case_variation.txt', 'Layer 1 Case Variation', '0 chunks stored, 4 duplicates (100% dedup)'),
                ('test_layer1_space_variation.txt', 'Layer 1 Space Variation', '0 chunks stored, 4 duplicates (100% dedup)'),
                ('test_layer1_punctuation_variation.txt', 'Layer 1 Punctuation', '0 chunks stored, 4 duplicates (100% dedup)'),
            ]
        },
        # Test Set 2: Layer 2 Testing
        {
            'name': 'TEST SET 2: Layer 2 (Semantic Similarity)',
            'files': [
                ('test_layer2_original.txt', 'Layer 2 Original', '3 chunks stored'),
                ('test_layer2_paraphrase.txt', 'Layer 2 Paraphrase', '0-1 chunks (66-100% dedup by Layer 2)'),
                ('test_layer2_similar.txt', 'Layer 2 Similar', '2-3 chunks stored'),
            ]
        },
        # Test Set 3: Layer 3 Testing
        {
            'name': 'TEST SET 3: Layer 3 (Cross-File Deduplication)',
            'files': [
                ('test_layer3_report_q1.txt', 'Layer 3 Q1 Report', '3 chunks stored'),
                ('test_layer3_report_q2.txt', 'Layer 3 Q2 Report', '1 chunk stored, 2 duplicates (66% dedup)'),
                ('test_layer3_report_q3.txt', 'Layer 3 Q3 Report', '1 chunk stored, 2 duplicates (66% dedup)'),
            ]
        },
        # Test Set 4: CSV Testing
        {
            'name': 'TEST SET 4: CSV File Deduplication',
            'files': [
                ('test_customers_jan.csv', 'CSV January', '3 rows stored'),
                ('test_customers_feb.csv', 'CSV February', '2 rows stored, 1 duplicate (33% dedup)'),
            ]
        },
        # Test Set 5: Mixed Content
        {
            'name': 'TEST SET 5: Mixed Content Deduplication',
            'files': [
                ('test_mixed_content_a.txt', 'Mixed Content A', '~7 chunks stored'),
                ('test_mixed_content_b.txt', 'Mixed Content B', '~4 chunks stored, ~3 duplicates (43% dedup)'),
            ]
        },
        # Test Set 6: Edge Cases
        {
            'name': 'TEST SET 6: Edge Cases',
            'files': [
                ('test_edge_empty.txt', 'Edge Case: Empty', '0 chunks'),
                ('test_edge_short.txt', 'Edge Case: Short', '0-1 chunks'),
                ('test_edge_long_repeated.txt', 'Edge Case: Repeated', '1 chunk, 99 duplicates (99% dedup!)'),
            ]
        },
    ]

    # Run all test sets
    success_count = 0
    total_count = 0

    for test_set in test_cases:
        print_header(test_set['name'])

        for filename, test_name, expected in test_set['files']:
            file_path = TEST_DIR / filename
            if file_path.exists():
                success = upload_file(file_path, test_name, expected)
                total_count += 1
                if success:
                    success_count += 1
                time.sleep(SLEEP_TIME)
            else:
                print_error(f"File not found: {filename}")

    # Final summary
    print_header("✅ ALL TESTS COMPLETED!")

    print(f"\n{Colors.GREEN}{'═' * 70}{Colors.ENDC}")
    print(f"{Colors.GREEN}Uploaded: {success_count}/{total_count} files{Colors.ENDC}")
    print(f"{Colors.GREEN}{'═' * 70}{Colors.ENDC}\n")

    print_info("📊 Now verify results:\n")
    print("1. Check logs:")
    print("   tail -f logs/ingestion.log | grep -E 'Dedup|Pipeline complete'")
    print("   (Windows: Get-Content logs/ingestion.log -Tail 50 -Wait)")
    print()
    print("2. Query database (PostgreSQL):")
    print("   psql -d marketing_advantage -f verify_dedup_results.sql")
    print()
    print("3. Or run Python verification:")
    print("   python verify_dedup_results.py")
    print()
    print(f"{Colors.BLUE}Expected overall deduplication: 70-75%{Colors.ENDC}")
    print()
    print(f"{Colors.GREEN}🎉 Testing suite execution complete!{Colors.ENDC}\n")

    return True

if __name__ == "__main__":
    try:
        success = run_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Testing interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
