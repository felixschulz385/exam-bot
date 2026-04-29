import argparse
import copy
import logging
import sys
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from src.config import ConfigLoader, DEFAULT_CONFIG_PATH
from src.data import QuestionBankReader, ExamSelectionWriter
from src.selection import UnifiedSampler
from src.utils import ensure_directory, resolve_path, timestamp_string

try:
    from src.generation import get_generator_class
except ImportError:
    get_generator_class = None

try:
    from src.generation.marking_sheet import MarkingSheetGenerator
except ImportError:
    MarkingSheetGenerator = None

try:
    from src.analysis.results_processor import ExamResultsProcessor
except ImportError:
    ExamResultsProcessor = None

SUPPORTED_SELECTION_METHODS = {"unified"}
DEFAULT_EXAM_CONFIG_PATH = DEFAULT_CONFIG_PATH


def get_document_generator_class(backend: str):
    """Load the configured document generator lazily."""
    global get_generator_class
    if get_generator_class is None:
        from src.generation import get_generator_class as loaded
        get_generator_class = loaded
    return get_generator_class(backend)


def get_marking_sheet_generator_class():
    """Load the marking-sheet generator lazily."""
    global MarkingSheetGenerator
    if MarkingSheetGenerator is None:
        from src.generation.marking_sheet import MarkingSheetGenerator as loaded
        MarkingSheetGenerator = loaded
    return MarkingSheetGenerator


def get_results_processor_class():
    """Load the results processor lazily."""
    global ExamResultsProcessor
    if ExamResultsProcessor is None:
        from src.analysis.results_processor import ExamResultsProcessor as loaded
        ExamResultsProcessor = loaded
    return ExamResultsProcessor

def setup_logging(config: Dict[str, Any], output_dir: Optional[Path] = None) -> None:
    """Set up logging based on configuration.
    
    Args:
        config: Configuration dictionary with logging settings
        output_dir: Optional output directory for log file
    """
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    log_file = log_config.get('file')
    
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    root_logger = logging.getLogger("")
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    if log_file:
        # If output_dir is provided, redirect log file there
        if output_dir:
            log_file_path = output_dir / Path(log_file).name
        else:
            log_file_path = Path(log_file)
            
        logging.basicConfig(
            level=numeric_level,
            format=log_format,
            filename=log_file_path,
            filemode='a',
            force=True
        )
        # Also log to console
        console = logging.StreamHandler()
        console.setLevel(numeric_level)
        console.setFormatter(logging.Formatter(log_format))
        logging.getLogger('').addHandler(console)
    else:
        logging.basicConfig(
            level=numeric_level,
            format=log_format,
            force=True
        )


def prepare_generation_config(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    """Resolve config-relative paths and validate generation settings."""
    prepared = copy.deepcopy(config)
    config_dir = config_path.resolve().parent

    io_config = prepared.setdefault("io", {})
    selection_config = prepared.setdefault("selection", {})
    semantic_config = selection_config.setdefault("semantic", {})

    for key in ("question_bank_path", "output_dir"):
        if key in io_config and io_config.get(key):
            io_config[key] = str(resolve_path(io_config[key], config_dir))

    if semantic_config.get("cache_dir"):
        semantic_config["cache_dir"] = str(resolve_path(semantic_config["cache_dir"], config_dir))

    block_question_config = prepared.get("document", {}).get("block_questions", {})
    if block_question_config.get("image_directory"):
        block_question_config["image_directory"] = str(
            resolve_path(block_question_config["image_directory"], config_dir)
        )

    method = selection_config.get("method", "unified")
    if method not in SUPPORTED_SELECTION_METHODS:
        raise ValueError(
            f"Unsupported selection method '{method}'. Supported methods: {sorted(SUPPORTED_SELECTION_METHODS)}"
        )

    question_bank_path = Path(io_config["question_bank_path"])
    if not question_bank_path.exists():
        raise FileNotFoundError(f"Question bank file not found: {question_bank_path}")

    target_points = selection_config.get("target_points")
    if target_points is not None and target_points <= 0:
        raise ValueError("selection.target_points must be positive when provided")

    total_questions = prepared.get("exam", {}).get("total_questions")
    if total_questions is not None and total_questions <= 0:
        raise ValueError("exam.total_questions must be positive when provided")

    return prepared

def select_questions(config: Dict[str, Any], question_bank: QuestionBankReader) -> list:
    """Select questions based on configuration settings.
    
    Args:
        config: Configuration dictionary
        question_bank: QuestionBankReader instance
        
    Returns:
        List of selected question indices
    """
    selection_config = config.get('selection', {})
    method = selection_config.get('method', 'unified')
    seed = selection_config.get('seed')
    target_points = selection_config.get('target_points', None)
    topics = selection_config.get('topics', [])
    topic_ratios = selection_config.get('topic_ratios', {})
    type_ratios = selection_config.get('type_ratios', {})
    
    # Get maximum questions per type (if configured)
    max_per_type = selection_config.get('max_per_type', {})
    
    grading_config = config.get('grading', {})
    points_per_type = grading_config.get('points_per_type', {})

    if target_points is None:
        raise ValueError("selection.target_points is required for exact type-ratio selection")

    logging.info(f"Using target points: {target_points} total exam points")
    
    questions_df = question_bank.get_all_questions()
    logging.info(f"Loaded {len(questions_df)} questions from the configured bank")
    
    if method == 'unified':
        semantic_config = selection_config.get('semantic', {})
        model = semantic_config.get('model', 'paraphrase-multilingual-MiniLM-L12-v2')
        cache_dir = semantic_config.get('cache_dir', 'cache')
        cluster_ratio = semantic_config.get('cluster_ratio', 0.3)
        
        sampler = UnifiedSampler(model_name=model, cache_dir=cache_dir, seed=seed)
        return sampler.select_unified(
            questions_df, 
            topic_ratios,
            type_ratios,
            points_per_type,
            target_points=target_points,
            max_per_type=max_per_type,
            cluster_ratio=cluster_ratio,
            question_bank_path=Path(config['io']['question_bank_path']),
            topics=topics,
        )
    else:
        raise ValueError(f"Unknown selection method: {method}")

def process_results(results_file: Path, marking_sheet: Path, output_dir: Path = None, output_name: str = None) -> None:
    """Process exam results based on student answers and marking sheet.
    
    Args:
        results_file: Path to Excel file with student answers
        marking_sheet: Path to marking sheet Excel file
        output_dir: Directory to save output files (optional)
        output_name: Base name for output files (optional)
    """
    results_file = results_file.expanduser().resolve()
    marking_sheet = marking_sheet.expanduser().resolve()

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    if not marking_sheet.exists():
        raise FileNotFoundError(f"Marking sheet not found: {marking_sheet}")

    logging.info(f"Processing exam results from: {results_file}")
    logging.info(f"Using marking sheet: {marking_sheet}")
    run_timestamp = timestamp_string()
    
    # Create output directory if not specified
    if output_dir is None:
        output_dir = Path(f"output/results_{run_timestamp}").resolve()
    else:
        output_dir = output_dir.expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set default output name if not specified
    if output_name is None:
        output_name = f"exam_results_{run_timestamp.split('_')[0]}"
    
    # Process results
    processor = get_results_processor_class()(output_dir=output_dir)
    
    try:
        # Load data
        results_df = processor.load_results(results_file)
        marking_df = processor.load_marking_sheet(marking_sheet)
        
        # Calculate scores
        student_scores = processor.calculate_student_scores(results_df, marking_df)
        
        # Generate question statistics
        question_stats = processor.generate_question_statistics(
            results_df, marking_df, student_scores
        )
        
        # Generate and save summary report
        output_path = processor.generate_summary_report(
            student_scores, question_stats, output_name
        )
        
        logging.info(f"Results processed successfully. Summary report saved to: {output_path}")
        
    except Exception as e:
        logging.error(f"Error processing results: {e}", exc_info=True)
        sys.exit(1)

def main(config_path: Path = None, results_file: Path = None, marking_sheet: Path = None, 
         results_output_dir: Path = None, results_output_name: str = None) -> None:
    """Main function to either generate an exam or process exam results.
    
    Args:
        config_path: Path to configuration file for exam generation
        results_file: Path to Excel file with student answers
        marking_sheet: Path to marking sheet Excel file
        results_output_dir: Directory to save results output files
        results_output_name: Base name for results output files
    """
    # Check if we're processing results
    if results_file and marking_sheet:
        process_results(results_file, marking_sheet, results_output_dir, results_output_name)
        return
    
    # If no config_path, show error and exit
    if not config_path:
        config_path = DEFAULT_EXAM_CONFIG_PATH
    
    # Load configuration
    config_path = config_path.expanduser().resolve()
    config_loader = ConfigLoader()
    config = prepare_generation_config(config_loader.load_config(config_path), config_path)
    
    # Create timestamped output directory
    timestamp = timestamp_string()
    base_output_dir = Path(config['io'].get('output_dir', 'output'))
    ensure_directory(base_output_dir)

    timestamp_output = config["io"].get("timestamp_output", True)
    output_dir = base_output_dir / f"exam_{timestamp}" if timestamp_output else base_output_dir
    ensure_directory(output_dir)
    
    # Update output directory in config for generators
    config['io']['output_dir'] = str(output_dir)
    
    # Set up logging to write to the new output directory
    setup_logging(config, output_dir)
    
    logging.info(f"Starting exam generation with config: {config_path}")
    logging.info(f"Created output directory: {output_dir}")
    
    try:
        # Load question bank
        question_bank_path = Path(config['io']['question_bank_path'])
        question_bank_sheet = config["io"].get("question_bank_sheet", QuestionBankReader.DEFAULT_SHEET_NAME)
        question_bank = QuestionBankReader(question_bank_path, sheet_name=question_bank_sheet)
        
        # Select questions
        selected_indices = select_questions(config, question_bank)
        
        # Mark selected questions in Excel file and save to output directory
        selection_info = {
            'selection_method': config['selection']['method'],
            'target_points': config['selection'].get('target_points'),
            'seed': config['selection'].get('seed'),
            'config_file': str(config_path)
        }
        
        # Create a writer that saves to our output directory
        writer = ExamSelectionWriter(
            question_bank_path,
            sheet_name=question_bank_sheet
        )
        excel_output_path = output_dir / f"selected_questions_{timestamp}.xlsx"
        marked_excel_path = writer.export_selected_questions(
            selected_indices, 
            selection_info,
            output_path=excel_output_path
        )
        
        # Get selected questions as DataFrame
        selected_questions = question_bank.get_questions_by_indices(selected_indices)
        
        # Pass question_bank to config for block information
        config['_runtime'] = {
            'question_bank_reader': question_bank
        }
        
        document_backend = config.get("document", {}).get("backend", "latex")
        generator = get_document_generator_class(document_backend)(output_dir=output_dir)
        
        # Get document outputs and answer mappings - REMOVED PDF paths
        student_path, instructor_path, answer_mappings = generator.generate_exam(
            selected_questions, config
        )
        
        # Generate marking sheet with answer mappings
        marking_generator = get_marking_sheet_generator_class()(output_dir=output_dir)
        marking_sheet_path = marking_generator.generate_marking_sheet(
            selected_questions, 
            config,
            timestamp=timestamp,
            answer_mappings=answer_mappings
        )
        
        # Save a copy of the config file used
        config_output_path = output_dir / f"config_{timestamp}.yaml"
        shutil.copy(config_path, config_output_path)
        
        logging.info("Exam generation completed successfully")
        logging.info(f"All outputs saved to: {output_dir}")
        if student_path:
            logging.info(f"Student version: {student_path}")
        if instructor_path:
            logging.info(f"Instructor version: {instructor_path}")
        # REMOVED: PDF logging section
        logging.info(f"Marking sheet: {marking_sheet_path}")
        logging.info(f"Selected questions: {marked_excel_path}")
        logging.info(f"Configuration: {config_output_path}")
            
    except Exception as e:
        logging.error(f"Error generating exam: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an exam from a question bank or process exam results")
    
    # Create mutually exclusive groups for different modes
    mode_group = parser.add_mutually_exclusive_group(required=False)
    
    # Exam generation mode
    mode_group.add_argument(
        '--config',
        type=Path,
        help=f"Path to configuration file for exam generation (defaults to {DEFAULT_EXAM_CONFIG_PATH})"
    )
    
    # Results processing mode
    mode_group.add_argument('--results', type=Path, help="Path to Excel file with student answers")
    
    # Options for results processing
    parser.add_argument('--marking-sheet', type=Path, help="Path to marking sheet Excel file (required with --results)")
    parser.add_argument('--results-output-dir', type=Path, help="Directory to save results output files")
    parser.add_argument('--results-output-name', type=str, help="Base name for results output files")
    
    args = parser.parse_args()
    
    # Configure basic logging for startup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Check if results mode is selected but marking sheet is missing
    if args.results and not args.marking_sheet:
        parser.error("--marking-sheet is required when using --results")
    
    # Call main function with appropriate arguments
    main(
        config_path=args.config,
        results_file=args.results,
        marking_sheet=args.marking_sheet,
        results_output_dir=args.results_output_dir,
        results_output_name=args.results_output_name
    )
