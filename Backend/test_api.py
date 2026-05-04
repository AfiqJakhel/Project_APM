"""
test_api.py
===========
Script untuk testing semua endpoint API prediksi harga cabai.
Jalankan setelah server running: uvicorn app.main:app --reload

Usage:
    python test_api.py
"""

import requests
import json
import sys
import io
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/predict"

def print_test(name, url, response):
    """Helper untuk print hasil test"""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"URL: {url}")
    print(f"Status: {response.status_code}")
    
    try:
        data = response.json()
        print(f"Response:\n{json.dumps(data, indent=2, ensure_ascii=False)}")
    except:
        print(f"Response (text): {response.text[:200]}")
    
    if response.status_code in [200, 201]:
        print("✅ PASS")
    else:
        print("❌ FAIL")
    
    return response.status_code in [200, 201]


def test_root():
    """Test root endpoint"""
    url = f"{BASE_URL}/"
    response = requests.get(url)
    return print_test("Root Endpoint", url, response)


def test_health():
    """Test health check"""
    url = f"{BASE_URL}{API_PREFIX}/health"
    response = requests.get(url)
    return print_test("Health Check", url, response)


def test_prediksi_h1():
    """Test prediksi H+1"""
    url = f"{BASE_URL}{API_PREFIX}/prediksi/h1"
    response = requests.get(url)
    return print_test("Prediksi H+1", url, response)


def test_prediksi_h3():
    """Test prediksi H+3"""
    url = f"{BASE_URL}{API_PREFIX}/prediksi/h3"
    response = requests.get(url)
    return print_test("Prediksi H+3", url, response)


def test_prediksi_h7():
    """Test prediksi H+7"""
    url = f"{BASE_URL}{API_PREFIX}/prediksi/h7"
    response = requests.get(url)
    return print_test("Prediksi H+7", url, response)


def test_prediksi_arah():
    """Test prediksi khusus arah H+1"""
    url = f"{BASE_URL}{API_PREFIX}/prediksi/arah/h1"
    response = requests.get(url)
    return print_test("Prediksi Arah H+1", url, response)


def test_prediksi_semua():
    """Test prediksi semua horizon"""
    url = f"{BASE_URL}{API_PREFIX}/prediksi"
    response = requests.get(url)
    return print_test("Prediksi Semua Horizon", url, response)


def test_harga_historis():
    """Test data historis"""
    url = f"{BASE_URL}{API_PREFIX}/harga/historis?n_hari=30"
    response = requests.get(url)
    return print_test("Data Historis (30 hari)", url, response)


def test_model_metrik():
    """Test metrik model"""
    url = f"{BASE_URL}{API_PREFIX}/model/metrik"
    response = requests.get(url)
    return print_test("Metrik Model", url, response)


def test_model_info():
    """Test info model"""
    url = f"{BASE_URL}{API_PREFIX}/model/info"
    response = requests.get(url)
    return print_test("Info Model", url, response)


def test_tanggal_tersedia():
    """Test rentang tanggal"""
    url = f"{BASE_URL}{API_PREFIX}/tanggal-tersedia"
    response = requests.get(url)
    return print_test("Tanggal Tersedia", url, response)


def test_fitur_terkini():
    """Test fitur terkini"""
    url = f"{BASE_URL}{API_PREFIX}/fitur-terkini"
    response = requests.get(url)
    return print_test("Fitur Terkini", url, response)


def test_cache_info():
    """Test cache info"""
    url = f"{BASE_URL}{API_PREFIX}/cache-info"
    response = requests.get(url)
    return print_test("Cache Info", url, response)


def test_invalid_horizon():
    """Test validasi horizon tidak valid"""
    url = f"{BASE_URL}{API_PREFIX}/prediksi/h2"
    response = requests.get(url)
    return print_test("Validasi Horizon Invalid (Expected 422)", url, response)


def main():
    """Run all tests"""
    print("="*70)
    print("TESTING API PREDIKSI HARGA CABAI")
    print(f"Base URL: {BASE_URL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*70)
    
    tests = [
        ("Root Endpoint", test_root),
        ("Health Check", test_health),
        ("Prediksi H+1", test_prediksi_h1),
        ("Prediksi H+3", test_prediksi_h3),
        ("Prediksi H+7", test_prediksi_h7),
        ("Prediksi Arah H+1", test_prediksi_arah),
        ("Prediksi Semua Horizon", test_prediksi_semua),
        ("Data Historis", test_harga_historis),
        ("Metrik Model", test_model_metrik),
        ("Info Model", test_model_info),
        ("Tanggal Tersedia", test_tanggal_tersedia),
        ("Fitur Terkini", test_fitur_terkini),
        ("Cache Info", test_cache_info),
        ("Validasi Horizon Invalid", test_invalid_horizon),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except requests.exceptions.ConnectionError:
            print(f"\n❌ FAIL: {name} - Server tidak running!")
            results.append((name, False))
        except Exception as e:
            print(f"\n❌ FAIL: {name} - Error: {e}")
            results.append((name, False))
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "✅ PASS" if p else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\n{'='*70}")
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("="*70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
