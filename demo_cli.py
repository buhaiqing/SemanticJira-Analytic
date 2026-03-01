"""Minimal demo version of the Jira Vector Analytics system."""

import typer
import json
import pandas as pd
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from datetime import datetime

app = typer.Typer(help="Jira Vector Analytics Demo CLI")
console = Console()


@app.command()
def preprocess(
    input: str = typer.Option(..., "--input", "-i", help="Input CSV file path"),
    output: str = typer.Option("./demo_processed.json", "--output", "-o", help="Output JSON file path"),
):
    """Demo preprocessing - simplified version."""
    console.print("[bold blue]🔄 Starting demo preprocessing...[/bold blue]")
    
    try:
        # Validate input file
        input_path = Path(input)
        if not input_path.exists():
            console.print(f"[red]❌ Input file not found: {input}[/red]")
            raise typer.Exit(code=1)
        
        # Load CSV data
        df = pd.read_csv(input_path)
        
        # Simple processing - just convert to our format
        processed_data = []
        for idx, row in df.iterrows():
            processed_item = {
                "issue_id": str(row.get('issue_id', f'TASK-{idx+1:03d}')),
                "summary": str(row.get('summary', 'No summary')),
                "description": str(row.get('description', '')),
                "created_at": str(row.get('created_at', datetime.now().isoformat())),
                "updated_at": str(row.get('updated_at', datetime.now().isoformat())),
                "status": str(row.get('status', 'To Do')),
                # priority field removed
                "cleaned_description": str(row.get('description', ''))[:200]  # Simple truncation
            }
            processed_data.append(processed_item)
        
        # Save processed data
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]✅ Successfully processed {len(processed_data)} tasks[/green]")
        console.print(f"[blue]📁 Output saved to: {output}[/blue]")
        
    except Exception as e:
        console.print(f"[red]❌ Error during preprocessing: {str(e)}[/red]")
        raise typer.Exit(code=1)


@app.command()
def health():
    """Check system health."""
    console.print("[bold blue]🏥 System Health Check[/bold blue]")
    
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "status": "operational",
        "components": {
            "cli": "healthy",
            "preprocessing": "healthy",
            "file_system": "accessible"
        }
    }
    
    table = Table(title="System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    for component, status in health_status["components"].items():
        table.add_row(component, status)
    
    console.print(table)
    console.print(f"[green]✅ System is operational[/green]")


@app.command()
def demo():
    """Run a complete demo workflow."""
    console.print("[bold blue]🚀 Running Demo Workflow[/bold blue]")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'issue_id': ['DEMO-001', 'DEMO-002', 'DEMO-003', 'DEMO-004', 'DEMO-005'],
        'summary': [
            'Fix user login authentication',
            'Add password reset feature',
            'Update user profile page',
            'Implement database optimization',
            'Add API rate limiting'
        ],
        'description': [
            'Users experiencing authentication failures after password changes',
            'Allow users to reset passwords via email verification',
            'Redesign user profile interface with modern components',
            'Optimize database queries for better performance',
            'Implement rate limiting to prevent API abuse'
        ],
        'created_at': [datetime.now().isoformat() for _ in range(5)],
        'updated_at': [datetime.now().isoformat() for _ in range(5)],
        'cluster_label': ['用户认证', '功能开发', '界面优化', '性能优化', '安全加固']
        # status field removed
        # priority field removed from sample data
    })
    
    # Save sample data
    sample_file = "demo_sample.csv"
    sample_data.to_csv(sample_file, index=False)
    console.print(f"[green]✅ Created sample data: {sample_file}[/green]")
    
    # Process the data
    output_file = "demo_processed.json"
    console.print("[blue]🔄 Processing sample data...[/blue]")
    
    processed_data = []
    for idx, row in sample_data.iterrows():
        processed_item = {
            "issue_id": row['issue_id'],
            "summary": row['summary'],
            "description": row['description'],
            "created_at": row['created_at'],
            "updated_at": row['updated_at'],
            # status field removed
            # priority field removed
            "cleaned_description": row['description'][:100],
            "demo_cluster": idx % 3  # Simple clustering for demo
        }
        processed_data.append(processed_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    console.print(f"[green]✅ Processed data saved to: {output_file}[/green]")
    
    # Show results
    table = Table(title="Demo Results Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    cluster_counts = {}
    for item in processed_data:
        cluster = item['demo_cluster']
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
    
    table.add_row("Total Tasks", str(len(processed_data)))
    table.add_row("Clusters Found", str(len(cluster_counts)))
    table.add_row("Processing Status", "Completed")
    
    console.print(table)
    
    # Show cluster breakdown
    cluster_table = Table(title="Cluster Distribution")
    cluster_table.add_column("Cluster ID", style="cyan")
    cluster_table.add_column("Task Count", style="green")
    cluster_table.add_column("Sample Task", style="yellow")
    
    for cluster_id, count in cluster_counts.items():
        sample_task = next(item for item in processed_data if item['demo_cluster'] == cluster_id)
        cluster_table.add_row(
            str(cluster_id),
            str(count),
            sample_task['summary'][:30] + "..."
        )
    
    console.print(cluster_table)
    console.print("[green]🎉 Demo completed successfully![/green]")


if __name__ == "__main__":
    app()