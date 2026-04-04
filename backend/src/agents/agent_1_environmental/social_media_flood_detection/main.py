"""
Social Media Flood Detection - Main Execution Script
=====================================================
Complete pipeline for processing social media data and detecting flood events.

Author: Mahmudul Hasan
Project: Autonomous Multi-Agent System for Urban Flood Response
Component: Agent 1 - Environmental Intelligence (Social Media)

Usage:
    python main.py                    # Run with default settings
    python main.py --generate-only    # Only generate dataset
    python main.py --process-only     # Process existing dataset
"""

import os
import sys
import json
import csv
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_CONFIG, DHAKA_ZONES
from sample_dataset_generator import generate_dataset, save_dataset
from openai_processor import (
    OpenAIFloodDetector,
    analyze_results,
    evaluate_accuracy,
    FloodDetectionResult
)


def print_banner() -> None:
    """Print application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     SOCIAL MEDIA FLOOD DETECTION SYSTEM                              ║
║     ──────────────────────────────────────                           ║
║     Agent 1 - Environmental Intelligence                             ║
║     Autonomous Multi-Agent Flood Response System                     ║
║                                                                      ║
║     University Capstone Project                                      ║
║     Author: Mahmudul Hasan                                           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def save_results_json(
    results: List[FloodDetectionResult],
    analysis: Dict[str, Any],
    evaluation: Dict[str, Any],
    cost: Dict[str, Union[int, float]],
    output_dir: str
) -> str:
    """Save complete results to JSON file"""
    
    output: Dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "system": "Social Media Flood Detection",
            "component": "Agent 1 - Environmental Intelligence",
            "model_used": "GPT-3.5-turbo"
        },
        "cost_analysis": cost,
        "detection_results": [r.to_dict() for r in results],
        "analysis_summary": analysis,
        "accuracy_evaluation": evaluation
    }
    
    filepath = os.path.join(output_dir, f"flood_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    return filepath


def save_results_csv(results: List[FloodDetectionResult], output_dir: str) -> str:
    """Save results to CSV for easy analysis"""
    
    filepath = os.path.join(output_dir, f"flood_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    fieldnames = [
        'tweet_id', 'is_flood_related', 'confidence', 'flood_severity',
        'location_mentioned', 'zone_detected', 'flood_type', 
        'urgency_level', 'needs_rescue', 'original_text'
    ]
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results:
            row: Dict[str, Any] = {
                'tweet_id': r.tweet_id,
                'is_flood_related': r.is_flood_related,
                'confidence': r.confidence,
                'flood_severity': r.flood_severity,
                'location_mentioned': r.location_mentioned,
                'zone_detected': r.zone_detected,
                'flood_type': r.flood_type,
                'urgency_level': r.urgency_level,
                'needs_rescue': r.needs_rescue,
                'original_text': r.original_text[:200]  # Truncate for CSV
            }
            writer.writerow(row)
    
    return filepath


def save_high_priority_alerts(results: List[FloodDetectionResult], output_dir: str) -> str:
    """Save high-priority alerts to separate file"""
    
    high_priority = [
        r for r in results 
        if r.needs_rescue or r.flood_severity in ['severe', 'critical']
    ]
    
    filepath = os.path.join(output_dir, f"high_priority_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    output: Dict[str, Any] = {
        "alert_type": "HIGH PRIORITY FLOOD ALERTS",
        "generated_at": datetime.now().isoformat(),
        "total_alerts": len(high_priority),
        "alerts": []
    }
    
    for r in high_priority:
        alert: Dict[str, Any] = {
            "priority": "RESCUE_NEEDED" if r.needs_rescue else "SEVERE_FLOOD",
            "tweet_id": r.tweet_id,
            "text": r.original_text,
            "severity": r.flood_severity,
            "location": r.location_mentioned,
            "zone": r.zone_detected,
            "confidence": r.confidence,
            "extracted_info": {
                "water_level": r.extracted_info.water_level_mentioned,
                "contact": r.extracted_info.contact_info
            }
        }
        output["alerts"].append(alert)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    return filepath


def print_summary_report(
    analysis: Dict[str, Any], 
    evaluation: Dict[str, Any], 
    cost: Dict[str, Union[int, float]]
) -> None:
    """Print formatted summary report to console"""
    
    print("\n" + "=" * 70)
    print("                    FLOOD DETECTION SUMMARY REPORT")
    print("=" * 70)
    
    summary = analysis.get("summary", {})
    
    print("\n📊 DETECTION STATISTICS")
    print("-" * 40)
    print(f"  Total tweets analyzed:     {summary.get('total_tweets_analyzed', 0)}")
    print(f"  Flood-related detected:    {summary.get('flood_related_tweets', 0)} ({summary.get('flood_detection_rate', 0)}%)")
    print(f"  Rescue situations:         {summary.get('rescue_situations', 0)}")
    print(f"  High priority alerts:      {summary.get('high_priority_alerts', 0)}")
    print(f"  Average confidence:        {summary.get('average_confidence', 0)}")
    
    distributions = analysis.get("distributions", {})
    zones = distributions.get("zones", {})
    
    print("\n📍 ZONE DISTRIBUTION")
    print("-" * 40)
    for zone, count in sorted(zones.items(), key=lambda x: -x[1]):
        zone_info = DHAKA_ZONES.get(zone, {})
        zone_name = zone_info.get("name", zone.title()) if isinstance(zone_info, dict) else zone.title()
        bar = "█" * (count * 2)
        print(f"  {zone_name:15} {count:3} {bar}")
    
    severity_dist = distributions.get("severity", {})
    
    print("\n⚠️  SEVERITY DISTRIBUTION")
    print("-" * 40)
    severity_order = ["critical", "severe", "moderate", "minor", "none"]
    emoji_map: Dict[str, str] = {"critical": "🔴", "severe": "🟠", "moderate": "🟡", "minor": "🟢", "none": "⚪"}
    for sev in severity_order:
        count = severity_dist.get(sev, 0)
        bar = "█" * (count * 2)
        emoji = emoji_map.get(sev, "")
        print(f"  {emoji} {sev.title():10} {count:3} {bar}")
    
    print("\n🎯 MODEL ACCURACY (vs Ground Truth)")
    print("-" * 40)
    if "error" not in evaluation:
        flood = evaluation.get("flood_detection", {})
        print(f"  Accuracy:    {flood.get('accuracy', 0)}%")
        print(f"  Precision:   {flood.get('precision', 0)}%")
        print(f"  Recall:      {flood.get('recall', 0)}%")
        print(f"  F1 Score:    {flood.get('f1_score', 0)}%")
        
        cm = flood.get("confusion_matrix", {})
        print("\n  Confusion Matrix:")
        print(f"    TP: {cm.get('true_positives', 0):3}  FP: {cm.get('false_positives', 0):3}")
        print(f"    FN: {cm.get('false_negatives', 0):3}  TN: {cm.get('true_negatives', 0):3}")
        
        rescue = evaluation.get("rescue_detection", {})
        print(f"\n  Rescue Detection:")
        print(f"    Precision: {rescue.get('precision', 0)}%  Recall: {rescue.get('recall', 0)}%")
    
    print("\n💰 API COST ANALYSIS")
    print("-" * 40)
    print(f"  API calls made:       {cost.get('api_calls', 0)}")
    print(f"  Input tokens:         {cost.get('input_tokens', 0):,}")
    print(f"  Output tokens:        {cost.get('output_tokens', 0):,}")
    print(f"  Total tokens:         {cost.get('total_tokens', 0):,}")
    print(f"  Estimated cost:       ${cost.get('estimated_cost_usd', 0):.4f}")
    
    print("\n" + "=" * 70)


def run_pipeline(
    api_key: Optional[str] = None,
    generate_data: bool = True,
    data_path: Optional[str] = None,
    output_dir: str = "output"
) -> Dict[str, Any]:
    """
    Run complete flood detection pipeline
    
    Args:
        api_key: OpenAI API key (or from environment)
        generate_data: Whether to generate new dataset
        data_path: Path to existing dataset (if not generating)
        output_dir: Directory for output files
    """
    
    print_banner()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Get or generate dataset
    print("\n" + "=" * 70)
    print("STEP 1: DATA PREPARATION")
    print("=" * 70)
    
    dataset: Dict[str, Any]
    actual_data_path: str
    
    if generate_data:
        print("\n📝 Generating sample Twitter dataset...")
        dataset = generate_dataset(
            n_flood_severe=15,
            n_flood_moderate=25,
            n_flood_minor=20,
            n_rescue=10,
            n_non_flood=30
        )
        actual_data_path = save_dataset(dataset, "data/sample_tweets.json")
    else:
        actual_data_path = data_path if data_path is not None else "data/sample_tweets.json"
        print(f"\n📂 Loading existing dataset from: {actual_data_path}")
        with open(actual_data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"   Loaded {len(dataset['data'])} tweets")
    
    tweets: List[Dict[str, Any]] = dataset["data"]
    ground_truth: List[Dict[str, Any]] = dataset.get("_ground_truth", [])
    
    # Step 2: Process with OpenAI
    print("\n" + "=" * 70)
    print("STEP 2: OPENAI FLOOD DETECTION")
    print("=" * 70)
    
    # Get API key from parameter or environment
    actual_api_key: str = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
    
    if not actual_api_key:
        print("\n❌ Error: OpenAI API key not found!")
        return {"error": "no_api_key"}
    
    detector = OpenAIFloodDetector(api_key=actual_api_key)
    results = detector.process_tweets(tweets)
    cost = detector.get_cost_estimate()
    
    print(f"\n✅ Processed {len(results)} tweets")
    print(f"   Estimated cost: ${cost.get('estimated_cost_usd', 0):.4f}")
    
    # Step 3: Analyze results
    print("\n" + "=" * 70)
    print("STEP 3: ANALYSIS & EVALUATION")
    print("=" * 70)
    
    analysis = analyze_results(results)
    
    # Evaluate against ground truth if available
    evaluation: Dict[str, Any] = {}
    if ground_truth:
        print("\n📊 Evaluating against ground truth...")
        evaluation = evaluate_accuracy(results, ground_truth)
    else:
        print("\n⚠️ No ground truth available for evaluation")
    
    # Step 4: Save outputs
    print("\n" + "=" * 70)
    print("STEP 4: SAVING OUTPUTS")
    print("=" * 70)
    
    json_path = save_results_json(results, analysis, evaluation, cost, output_dir)
    print(f"\n📄 Full results saved to: {json_path}")
    
    csv_path = save_results_csv(results, output_dir)
    print(f"📄 CSV export saved to: {csv_path}")
    
    alerts_path = save_high_priority_alerts(results, output_dir)
    print(f"🚨 High priority alerts saved to: {alerts_path}")
    
    # Step 5: Print summary
    print_summary_report(analysis, evaluation, cost)
    
    # Return paths for external use
    return {
        "results_json": json_path,
        "results_csv": csv_path,
        "alerts_json": alerts_path,
        "analysis": analysis,
        "evaluation": evaluation,
        "cost": cost
    }


def main() -> None:
    """Main entry point with CLI argument parsing"""
    
    parser = argparse.ArgumentParser(
        description="Social Media Flood Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Full pipeline with new dataset
  python main.py --generate-only          # Only generate sample dataset
  python main.py --data data/tweets.json  # Use existing dataset
  python main.py --api-key YOUR_KEY       # Specify API key
        """
    )
    
    parser.add_argument(
        '--api-key', '-k',
        type=str,
        default=None,
        help='OpenAI API key (or set OPENAI_API_KEY env var)'
    )
    
    parser.add_argument(
        '--generate-only', '-g',
        action='store_true',
        help='Only generate dataset, do not process'
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default=None,
        help='Path to existing dataset file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output directory (default: output)'
    )
    
    args = parser.parse_args()
    
    if args.generate_only:
        print_banner()
        print("\n📝 Generating sample dataset only...")
        dataset = generate_dataset()
        save_dataset(dataset, "data/sample_tweets.json")
        print("\n✅ Dataset generation complete!")
        return
    
    # Get API key
    api_key: Optional[str] = args.api_key if args.api_key else os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("\n❌ Error: OpenAI API key required!")
        print("   Set OPENAI_API_KEY environment variable or use --api-key flag")
        sys.exit(1)
    
    # Run pipeline
    run_pipeline(
        api_key=api_key,
        generate_data=(args.data is None),
        data_path=args.data,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()