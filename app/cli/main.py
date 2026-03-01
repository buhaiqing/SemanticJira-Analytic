"""Main CLI application for Jira Vector Analytics."""

import typer
import asyncio
import logging
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from datetime import datetime

from app.core.preprocessing import DataPreprocessor
from app.core.embedding import VectorEmbedder
from app.core.clustering import TaskClusterer, ClusteringConfig
from app.core.conversation import ConversationalAgent

app = typer.Typer(help="Jira Vector Analytics CLI Tool")
console = Console()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.command()
def preprocess(
    input: str = typer.Option(..., "--input", "-i", help="Input CSV/Excel file path"),
    output: str = typer.Option("./processed_data.json", "--output", "-o", help="Output JSON file path"),
):
    """Preprocess Jira data for analysis."""
    console.print("[bold blue]🔄 Starting data preprocessing...[/bold blue]")
    
    try:
        # Validate input file
        input_path = Path(input)
        if not input_path.exists():
            console.print(f"[red]❌ Input file not found: {input}[/red]")
            raise typer.Exit(code=1)
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            # Load data
            load_task = progress.add_task("Loading data...", total=None)
            tasks = preprocessor.load_data(str(input_path))
            progress.update(load_task, completed=True)
            
            if not tasks:
                console.print("[yellow]⚠️  No valid tasks found in input file[/yellow]")
                return
            
            # Preprocess tasks
            preprocess_task = progress.add_task("Preprocessing tasks...", total=None)
            processed_tasks = preprocessor.preprocess_tasks(tasks)
            progress.update(preprocess_task, completed=True)
            
            # Save processed data
            save_task = progress.add_task("Saving processed data...", total=None)
            preprocessor.save_processed_data(processed_tasks, output)
            progress.update(save_task, completed=True)
        
        console.print(f"[green]✅ Successfully processed {len(processed_tasks)} tasks[/green]")
        console.print(f"[blue]📁 Output saved to: {output}[/blue]")
        
    except Exception as e:
        console.print(f"[red]❌ Error during preprocessing: {str(e)}[/red]")
        logger.error(f"Preprocessing error: {e}")
        raise typer.Exit(code=1)


@app.command()
def vectorize(
    input: str = typer.Option(..., "--input", "-i", help="Input processed JSON file"),
    output: str = typer.Option("./embedded_data.json", "--output", "-o", help="Output JSON file"),
    model: str = typer.Option("BGE-M3", "--model", "-m", help="Embedding model to use"),
):
    """Generate vector embeddings for processed tasks."""
    console.print(f"[bold blue]🧠 Generating embeddings with {model}...[/bold blue]")
    
    try:
        # Validate input file
        input_path = Path(input)
        if not input_path.exists():
            console.print(f"[red]❌ Input file not found: {input}[/red]")
            raise typer.Exit(code=1)
        
        # Load processed data
        preprocessor = DataPreprocessor()
        with console.status("[bold green]Loading processed data...") as status:
            tasks = preprocessor.load_processed_data(str(input_path))
            status.update("[bold green]Initializing embedding model...")
            
            # Initialize embedder
            embedder = VectorEmbedder(model_name=model)
            asyncio.run(embedder.initialize_model())
            
            status.update("[bold green]Generating embeddings...")
            # Generate embeddings
            embedded_tasks = asyncio.run(embedder.embed_tasks(tasks))
            
            # Cleanup
            asyncio.run(embedder.close())
        
        # Save embedded data
        with console.status("[bold green]Saving embedded data...") as status:
            preprocessor.save_processed_data(embedded_tasks, output)
        
        console.print(f"[green]✅ Generated embeddings for {len(embedded_tasks)} tasks[/green]")
        console.print(f"[blue]📁 Output saved to: {output}[/blue]")
        
    except Exception as e:
        console.print(f"[red]❌ Error during vectorization: {str(e)}[/red]")
        logger.error(f"Vectorization error: {e}")
        raise typer.Exit(code=1)


@app.command()
def cluster(
    input: str = typer.Option(..., "--input", "-i", help="Input embedded JSON file"),
    output: str = typer.Option("./cluster_results.json", "--output", "-o", help="Output results file"),
    algorithm: str = typer.Option("hdbscan", "--algorithm", "-a", help="Clustering algorithm"),
    min_size: int = typer.Option(10, "--min-size", help="Minimum cluster size"),
    epsilon: float = typer.Option(0.5, "--epsilon", help="Cluster selection epsilon"),
):
    """Perform clustering analysis on embedded tasks."""
    console.print(f"[bold blue]📊 Running {algorithm.upper()} clustering...[/bold blue]")
    
    try:
        # Validate input file
        input_path = Path(input)
        if not input_path.exists():
            console.print(f"[red]❌ Input file not found: {input}[/red]")
            raise typer.Exit(code=1)
        
        # Load embedded data
        preprocessor = DataPreprocessor()
        with console.status("[bold green]Loading embedded data...") as status:
            tasks = preprocessor.load_processed_data(str(input_path))
            
            # Configure clustering
            config = ClusteringConfig(
                algorithm=algorithm,
                min_cluster_size=min_size,
                cluster_selection_epsilon=epsilon
            )
            
            status.update("[bold green]Performing clustering analysis...")
            # Perform clustering
            clusterer = TaskClusterer(config)
            result = asyncio.run(clusterer.cluster_tasks(tasks))
            
            # Cleanup
            asyncio.run(clusterer.close())
        
        # Display results
        _display_clustering_results(result)
        
        # Save results
        with console.status("[bold green]Saving results...") as status:
            import json
            result_dict = result.model_dump()
            # Convert datetime to string for JSON serialization
            result_dict['generated_at'] = result_dict['generated_at'].isoformat()
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        console.print("[green]✅ Clustering completed successfully[/green]")
        console.print(f"[blue]📁 Results saved to: {output}[/blue]")
        
    except Exception as e:
        console.print(f"[red]❌ Error during clustering: {str(e)}[/red]")
        logger.error(f"Clustering error: {e}")
        raise typer.Exit(code=1)


@app.command()
def export(
    input: str = typer.Option(..., "--input", "-i", help="Input results JSON file"),
    output: str = typer.Option("./exported_results.csv", "--output", "-o", help="Output file path"),
    format: str = typer.Option("csv", "--format", "-f", help="Export format (csv/json)"),
):
    """Export analysis results to various formats."""
    console.print(f"[bold blue]📤 Exporting results as {format.upper()}...[/bold blue]")
    
    try:
        # Validate input file
        input_path = Path(input)
        if not input_path.exists():
            console.print(f"[red]❌ Input file not found: {input}[/red]")
            raise typer.Exit(code=1)
        
        # Load results
        import json
        with open(input_path, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        if format.lower() == "csv":
            _export_to_csv(result_data, output)
        elif format.lower() == "json":
            _export_to_json(result_data, output)
        else:
            console.print(f"[red]❌ Unsupported format: {format}[/red]")
            raise typer.Exit(code=1)
        
        console.print("[green]✅ Results exported successfully[/green]")
        console.print(f"[blue]📁 Output saved to: {output}[/blue]")
        
    except Exception as e:
        console.print(f"[red]❌ Error during export: {str(e)}[/red]")
        logger.error(f"Export error: {e}")
        raise typer.Exit(code=1)


@app.command()
def chat():
    """Start conversational agent for natural language analysis."""
    console.print("[bold blue]🤖 Starting conversational agent...[/bold blue]")
    
    try:
        # Run interactive chat loop
        asyncio.run(_run_chat_interface())
        
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Chat session ended by user[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Error in chat interface: {str(e)}[/red]")
        logger.error(f"Chat interface error: {e}")
        raise typer.Exit(code=1)


async def _run_chat_interface():
    """Run the interactive chat interface."""
    agent = ConversationalAgent()
    session_id = await agent.create_session()
    
    console.print("\n[green]✅ Agent initialized! Type 'help' for available commands.[/green]")
    console.print("[blue]💡 Tip: You can say things like '分析Jira任务' or '查看结果'[/blue]\n")
    
    while True:
        try:
            # Get user input
            user_input = console.input("[bold cyan]You:[/bold cyan] ")
            
            if user_input.lower() in ['quit', 'exit', '退出', '再见']:
                break
            
            if not user_input.strip():
                continue
            
            # Process message
            with console.status("[bold green]Thinking..."):
                response = await agent.process_message(session_id, user_input)
            
            # Display response
            console.print(f"[bold magenta]Agent:[/bold magenta] {response.message}\n")
            
            # Handle special actions
            if response.action_required == "file_upload":
                file_path = console.input("[bold yellow]请输入文件路径:[/bold yellow] ")
                if file_path:
                    # Simulate file upload
                    agent.sessions[session_id].context["uploaded_file_path"] = file_path
                    agent.sessions[session_id].context["file_uploaded"] = True
                    
                    # Process the file
                    processing_response = await agent.process_message(session_id, "已上传文件")
                    console.print(f"[bold magenta]Agent:[/bold magenta] {processing_response.message}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error processing message: {e}[/red]")
            continue
    
    # Cleanup
    await agent.cleanup_session(session_id)
    await agent.close()


@app.command()
def health():
    """Check system health and configuration."""
    console.print("[bold blue]🏥 Checking system health...[/bold blue]")
    
    try:
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Check required dependencies
        dependencies = ["numpy", "pandas", "sklearn", "hdbscan", "sentence_transformers"]
        for dep in dependencies:
            try:
                __import__(dep)
                health_status["components"][dep] = "healthy"
            except ImportError:
                health_status["components"][dep] = "missing"
        
        # Check data directory
        data_dir = Path("./data")
        if data_dir.exists():
            health_status["components"]["data_directory"] = "accessible"
        else:
            health_status["components"]["data_directory"] = "missing"
        
        # Display health report
        table = Table(title="System Health Report")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        
        for component, status in health_status["components"].items():
            table.add_row(component, status)
        
        console.print(table)
        console.print(f"[green]✅ Health check completed at {health_status['timestamp']}[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Health check failed: {str(e)}[/red]")
        logger.error(f"Health check error: {e}")
        raise typer.Exit(code=1)


def _display_clustering_results(result):
    """Display clustering results in a formatted table."""
    table = Table(title=f"Clustering Results ({result.total_tasks} tasks)")
    table.add_column("Cluster ID", style="cyan")
    table.add_column("Size", style="magenta")
    table.add_column("Avg Confidence", style="green")
    table.add_column("Sample Tasks", style="yellow")
    
    for cluster_id, details in result.cluster_details.items():
        sample_tasks = ", ".join([t["summary"] for t in details.get("sample_tasks", [])[:2]])
        table.add_row(
            str(cluster_id),
            str(details["size"]),
            f"{details['avg_confidence']:.3f}",
            sample_tasks[:50] + "..." if len(sample_tasks) > 50 else sample_tasks
        )
    
    console.print(table)
    console.print(f"[bold]Total Clusters:[/bold] {result.clusters_found}")
    console.print(f"[bold]Processing Time:[/bold] {result.processing_time:.2f}s")


def _export_to_csv(result_data: dict, output_path: str):
    """Export results to CSV format."""
    import pandas as pd
    
    # Flatten the cluster details for CSV export
    rows = []
    for cluster_id, details in result_data["cluster_details"].items():
        row = {
            "cluster_id": cluster_id,
            "size": details["size"],
            "avg_confidence": details["avg_confidence"],
            "is_noise": details.get("is_noise", False)
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8')


def _export_to_json(result_data: dict, output_path: str):
    """Export results to JSON format."""
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    app()