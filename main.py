import argparse
import logging
import sys
import shutil
import datetime
from pathlib import Path
from typing import Dict, Any

from src.config import ConfigLoader
from src.data import QuestionBankReader, ExamSelectionWriter, QuestionEmbedder
from src.selection import UnifiedSampler
from src.generation import WordGenerator, TemplateProcessor
from src.generation.marking_sheet import MarkingSheetGenerator

def setup_logging(config: Dict[str, Any], output_dir: Path = None) -> None:
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
    
    if log_file:
        # If output_dir is provided, redirect log file there
        if output_dir:
            log_file_path = output_dir / Path(log_file).name
        else:
            log_file_path = log_file
            
        logging.basicConfig(
            level=numeric_level,
            format=log_format,
            filename=log_file_path,
            filemode='a'
        )
        # Also log to console
        console = logging.StreamHandler()
        console.setLevel(numeric_level)
        console.setFormatter(logging.Formatter(log_format))
        logging.getLogger('').addHandler(console)
    else:
        logging.basicConfig(
            level=numeric_level,
            format=log_format
        )

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
    dynamic_sizing = selection_config.get('dynamic_sizing', False)
    target_points = selection_config.get('target_points', None)
    
    # Get maximum questions per type (if configured)
    max_per_type = selection_config.get('max_per_type', {})
    
    grading_config = config.get('grading', {})
    topic_points = grading_config.get('topic_points', {})
    type_points = grading_config.get('type_points', {})
    
    exam_config = config.get('exam', {})
    total_questions = None  # Will be set based on configuration
    
    # Determine how many questions to select based on config
    if target_points:
        # Use target_points as the main criterion
        logging.info(f"Using target points: {target_points} total exam points")
        # total_questions will be determined by the sampler based on points
        total_questions = None
    elif dynamic_sizing:
        # Sum all topic points to determine total questions
        total_questions = sum(topic_points.values())
        logging.info(f"Using dynamic sizing: {total_questions} questions based on topic points sum")
    else:
        # Fall back to fixed number of questions
        total_questions = exam_config.get('total_questions', 30)
        logging.info(f"Using fixed exam size: {total_questions} questions")
    
    # Get all questions and filter for those checked by Felix
    questions_df = question_bank.get_all_questions()
    if 'Check: Felix' in questions_df.columns:
        # Filter for questions that have been checked by Felix (value is True or 1)
        felix_checked = questions_df['Check: Felix'].fillna(False).astype(bool)
        questions_df = questions_df[felix_checked].copy()
        logging.info(f"Filtered to {len(questions_df)} questions checked by Felix")
    
    if method == 'unified':
        semantic_config = selection_config.get('semantic', {})
        model = semantic_config.get('model', 'paraphrase-multilingual-MiniLM-L12-v2')
        cache_dir = semantic_config.get('cache_dir', 'cache')
        cluster_ratio = semantic_config.get('cluster_ratio', 0.3)
        
        sampler = UnifiedSampler(model_name=model, cache_dir=cache_dir, seed=seed)
        return sampler.select_unified(
            questions_df, 
            topic_points, 
            type_points,
            total_questions=total_questions,
            target_points=target_points,
            max_per_type=max_per_type,
            dynamic_sizing=dynamic_sizing,
            cluster_ratio=cluster_ratio,
            question_bank_path=Path(config['io']['question_bank_path'])
        )
    else:
        raise ValueError(f"Unknown selection method: {method}")

def main(config_path: Path) -> None:
    """Main function to generate an exam based on configuration.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config(config_path)
    
    # Create timestamped output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(config['io'].get('output_dir', 'output'))
    if not base_output_dir.exists():
        base_output_dir.mkdir(parents=True)
        
    # Create timestamped subdirectory
    output_dir = base_output_dir / f"exam_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    
    # Update output directory in config for generators
    config['io']['output_dir'] = str(output_dir)
    
    # Set up logging to write to the new output directory
    setup_logging(config, output_dir)
    
    logging.info(f"Starting exam generation with config: {config_path}")
    logging.info(f"Created output directory: {output_dir}")
    
    try:
        # Load question bank
        question_bank_path = Path(config['io']['question_bank_path'])
        question_bank = QuestionBankReader(question_bank_path)
        
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
        writer = ExamSelectionWriter(question_bank_path, selection_column='2025 Exam')
        excel_output_path = output_dir / f"selected_questions_{timestamp}.xlsx"
        marked_excel_path = writer.mark_selected_questions(
            selected_indices, 
            selection_info,
            output_path=excel_output_path
        )
        
        # Get selected questions as DataFrame
        selected_questions = question_bank.get_questions_by_indices(selected_indices)
        
        # Generate exam documents
        template_path = config['io'].get('template_path')
        
        # Pass question_bank to config for block information
        config['_runtime'] = {
            'question_bank_reader': question_bank
        }
        
        generator = WordGenerator(
            template_path=Path(template_path) if template_path else None,
            output_dir=output_dir
        )
        
        # Get document outputs and answer mappings - REMOVED PDF paths
        student_path, instructor_path, answer_mappings = generator.generate_exam(
            selected_questions, config
        )
        
        # Generate marking sheet with answer mappings
        marking_generator = MarkingSheetGenerator(output_dir=output_dir)
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
    parser = argparse.ArgumentParser(description="Generate an exam from a question bank")
    parser.add_argument('config', type=Path, help="Path to configuration file (YAML or JSON)")
    args = parser.parse_args()
    
    main(args.config)