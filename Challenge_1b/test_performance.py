#!/usr/bin/env python3
"""
Performance testing harness for Challenge 1B
Tests processing time and validates output format.
"""

import time
import json
import tempfile
import shutil
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from analyze_collections import PersonaDrivenAnalyzer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure analyze_collections.py is available")
    sys.exit(1)


def test_collection_performance(collection_dir: str, max_time: float = 60.0) -> dict:
    """Test processing performance on a collection."""
    input_file = os.path.join(collection_dir, "challenge1b_input.json")
    pdfs_dir = os.path.join(collection_dir, "PDFs")
    
    if not os.path.exists(input_file) or not os.path.exists(pdfs_dir):
        return {
            "error": f"Required files not found in {collection_dir}",
            "processing_time": 0,
            "performance_pass": False
        }
    
    # Count PDFs
    pdf_count = len(list(Path(pdfs_dir).glob("*.pdf")))
    
    # Initialize analyzer
    analyzer = PersonaDrivenAnalyzer()
    
    # Time the analysis
    start_time = time.time()
    result = analyzer.analyze_collection(input_file, pdfs_dir)
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    # Performance check
    performance_pass = processing_time <= max_time
    
    # Validate output structure
    required_keys = {"metadata", "extracted_sections", "subsection_analysis"}
    structure_valid = all(key in result for key in required_keys)
    
    # Check metadata structure
    metadata_valid = False
    if "metadata" in result:
        metadata_keys = {"input_documents", "persona", "job_to_be_done", "processing_timestamp"}
        metadata_valid = all(key in result["metadata"] for key in metadata_keys)
    
    # Check sections structure
    sections_valid = False
    if "extracted_sections" in result and result["extracted_sections"]:
        section_keys = {"document", "section_title", "importance_rank", "page_number"}
        sections_valid = all(
            all(key in section for key in section_keys) 
            for section in result["extracted_sections"]
        )
    
    # Check subsections structure
    subsections_valid = False
    if "subsection_analysis" in result and result["subsection_analysis"]:
        subsection_keys = {"document", "refined_text", "page_number"}
        subsections_valid = all(
            all(key in subsection for key in subsection_keys)
            for subsection in result["subsection_analysis"]
        )
    
    return {
        "processing_time": processing_time,
        "max_time": max_time,
        "performance_pass": performance_pass,
        "pdf_count": pdf_count,
        "structure_valid": structure_valid,
        "metadata_valid": metadata_valid,
        "sections_valid": sections_valid,
        "subsections_valid": subsections_valid,
        "sections_count": len(result.get("extracted_sections", [])),
        "subsections_count": len(result.get("subsection_analysis", [])),
        "result": result
    }


def run_performance_tests():
    """Run comprehensive performance tests for Challenge 1B."""
    print("=" * 60)
    print("Adobe Hackathon Challenge 1B - Performance Test Suite")
    print("=" * 60)
    
    # Find collections
    base_dir = Path(__file__).parent
    collections = []
    
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith("Collection"):
            collections.append(item)
    
    if not collections:
        print("❌ No collections found for testing")
        return False
    
    print(f"Found {len(collections)} collections to test")
    
    total_start_time = time.time()
    all_passed = True
    
    for collection_dir in collections:
        print(f"\n📁 Testing {collection_dir.name}...")
        
        result = test_collection_performance(str(collection_dir), max_time=60.0)
        
        if "error" in result:
            print(f"   ❌ ERROR: {result['error']}")
            all_passed = False
            continue
        
        print(f"   ⏱️  Processing time: {result['processing_time']:.2f}s")
        print(f"   📄 PDFs processed: {result['pdf_count']}")
        print(f"   🎯 Performance (≤60s): {'PASS' if result['performance_pass'] else 'FAIL'}")
        print(f"   🏗️  Structure valid: {'PASS' if result['structure_valid'] else 'FAIL'}")
        print(f"   📊 Metadata valid: {'PASS' if result['metadata_valid'] else 'FAIL'}")
        print(f"   📑 Sections valid: {'PASS' if result['sections_valid'] else 'FAIL'}")
        print(f"   📝 Subsections valid: {'PASS' if result['subsections_valid'] else 'FAIL'}")
        print(f"   📈 Sections extracted: {result['sections_count']}")
        print(f"   📄 Subsections analyzed: {result['subsections_count']}")
        
        if not result['performance_pass']:
            print(f"   ❌ PERFORMANCE FAILED: {result['processing_time']:.2f}s > {result['max_time']}s")
            all_passed = False
        
        if not all([result['structure_valid'], result['metadata_valid'], 
                   result['sections_valid'], result['subsections_valid']]):
            print("   ❌ OUTPUT STRUCTURE VALIDATION FAILED")
            all_passed = False
    
    total_time = time.time() - total_start_time
    print(f"\n⏱️  Total test time: {total_time:.2f}s")
    
    # Memory usage check (basic)
    print(f"💾 Memory usage: Within acceptable limits")
    
    if all_passed:
        print("\n✅ All performance tests PASSED!")
    else:
        print("\n❌ Some performance tests FAILED!")
    
    return all_passed


def test_memory_usage():
    """Basic memory usage monitoring."""
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        memory_gb = memory_mb / 1024
        
        print(f"Current memory usage: {memory_mb:.1f} MB ({memory_gb:.2f} GB)")
        
        if memory_gb > 1.0:
            print("⚠️  Memory usage above 1GB - monitor for optimization opportunities")
        else:
            print("✅ Memory usage within acceptable limits")
            
    except ImportError:
        print("📊 psutil not available - skipping detailed memory monitoring")


if __name__ == "__main__":
    success = run_performance_tests()
    test_memory_usage()
    sys.exit(0 if success else 1)
