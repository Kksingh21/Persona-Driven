#!/usr/bin/env python3
"""
Test script to check Challenge 1B outputs for specific collections
"""
import sys
import os
import json
from pathlib import Path

# Add Challenge_1a to path for dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Challenge_1a'))
sys.path.append('.')

from analyze_collections import PersonaDrivenAnalyzer

def test_specific_collection(collection_name):
    """Test Challenge 1B with a specific collection."""
    print(f"=== TESTING CHALLENGE 1B with {collection_name} ===")
    
    collection_dir = collection_name
    input_file = os.path.join(collection_dir, "challenge1b_input.json")
    pdfs_dir = os.path.join(collection_dir, "PDFs")
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    if not os.path.exists(pdfs_dir):
        print(f"❌ PDFs directory not found: {pdfs_dir}")
        return
    
    # Load input configuration
    with open(input_file, 'r') as f:
        input_config = json.load(f)
    
    print(f"👤 Persona: {input_config['persona']['role']}")
    print(f"🎯 Task: {input_config['job_to_be_done']['task']}")
    print(f"📚 Documents: {len(input_config['documents'])} PDFs")
    
    # List PDFs
    pdf_files = list(Path(pdfs_dir).glob("*.pdf"))
    print(f"📄 Available PDFs:")
    for pdf in pdf_files:
        print(f"   - {pdf.name}")
    
    # Initialize analyzer
    analyzer = PersonaDrivenAnalyzer()
    
    # Analyze collection
    print(f"\n⏳ Analyzing collection...")
    result = analyzer.analyze_collection(input_file, pdfs_dir)
    
    # Show results
    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"⏱️  Processing timestamp: {result['metadata']['processing_timestamp']}")
    print(f"📑 Top sections extracted: {len(result['extracted_sections'])}")
    print(f"📝 Subsections analyzed: {len(result['subsection_analysis'])}")
    
    print(f"\n🏆 TOP RELEVANT SECTIONS:")
    for i, section in enumerate(result['extracted_sections'][:3], 1):
        print(f"{i}. \"{section['section_title']}\"")
        print(f"   📄 Document: {section['document']}")
        print(f"   📄 Page: {section['page_number']}")
        print(f"   ⭐ Rank: {section['importance_rank']}")
    
    print(f"\n📖 SAMPLE CONTENT ANALYSIS:")
    for i, subsection in enumerate(result['subsection_analysis'][:2], 1):
        print(f"{i}. From {subsection['document']} (Page {subsection['page_number']}):")
        content = subsection['refined_text']
        preview = content[:200] + "..." if len(content) > 200 else content
        print(f"   \"{preview}\"")
    
    # Save output
    output_file = f"test_output_{collection_name.replace(' ', '_').lower()}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Full output saved to: {output_file}")
    return result

if __name__ == "__main__":
    # Test different collections
    collections = ["Collection 1", "Collection 2", "Collection 3"]
    
    for collection in collections:
        if os.path.exists(collection):
            test_specific_collection(collection)
            print("\n" + "="*80 + "\n")
        else:
            print(f"❌ Collection not found: {collection}")
