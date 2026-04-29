import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, List
import argparse
from src.utils import ensure_directory

logger = logging.getLogger(__name__)

class ExamResultsProcessor:
    """Process exam results and generate summary statistics."""
    
    def __init__(self, output_dir: Path = Path("output")):
        """Initialize the exam results processor.
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = ensure_directory(Path(output_dir))
        logger.info(f"Using output directory: {self.output_dir}")
    
    def load_results(self, results_path: Path) -> pd.DataFrame:
        """Load student exam results from Excel file.
        
        Args:
            results_path: Path to Excel file with student answers
            
        Returns:
            DataFrame with student results
        """
        try:
            results_df = pd.read_excel(results_path)
            logger.info(f"Loaded results from {results_path}")
            
            # Handle the case where the first column is labeled "Question" and contains row numbers
            if results_df.columns[0] == "Question":
                # Make sure the question column contains integers
                results_df["Question"] = results_df["Question"].astype(int)
                
                # If the values start from 0, we might need to add 1 to match marking sheet
                # Uncomment the following line if question numbers should start from 1 instead of 0
                # results_df["Question"] = results_df["Question"] + 1
                
            return results_df
        except Exception as e:
            logger.error(f"Error loading results file: {e}")
            raise
    
    def load_marking_sheet(self, marking_sheet_path: Path) -> pd.DataFrame:
        """Load marking sheet with correct answers.
        
        Args:
            marking_sheet_path: Path to marking sheet Excel file
            
        Returns:
            DataFrame with correct answers and points
        """
        try:
            # Try to load the marking sheet
            try:
                # First try with the "Marking Sheet" sheet name
                marking_df = pd.read_excel(marking_sheet_path, sheet_name="Marking Sheet")
            except Exception:
                # If that fails, try the default sheet (first sheet)
                marking_df = pd.read_excel(marking_sheet_path)
                logger.info("Used default sheet instead of 'Marking Sheet'")
            
            # Check for expected columns
            expected_columns = ["Question #", "Correct Answer Letter", "Points"]
            for col in expected_columns:
                if col not in marking_df.columns:
                    # Try to find similar columns
                    if "Question" in marking_df.columns and col == "Question #":
                        marking_df = marking_df.rename(columns={"Question": "Question #"})
                    elif col == "Correct Answer Letter":
                        candidate_columns = [
                            column_name for column_name in marking_df.columns
                            if "answer" in str(column_name).lower() or "key" in str(column_name).lower()
                        ]
                        if candidate_columns:
                            marking_df = marking_df.rename(
                                columns={candidate_columns[0]: "Correct Answer Letter"}
                            )
                    else:
                        raise ValueError(f"Required column '{col}' not found in marking sheet")
            
            # Filter out the total row
            if "Total" in marking_df["Question #"].astype(str).values:
                marking_df = marking_df[marking_df["Question #"].astype(str) != "Total"]
            
            # Convert question numbers to integers if they're not already
            if marking_df["Question #"].dtype != int:
                # Handle cases where question numbers might be strings with "Question" prefix
                try:
                    marking_df["Question #"] = marking_df["Question #"].astype(int)
                except ValueError:
                    # Try to extract numbers from strings like "Question 1"
                    import re
                    marking_df["Question #"] = marking_df["Question #"].apply(
                        lambda x: int(re.search(r'\d+', str(x)).group()) if re.search(r'\d+', str(x)) else x
                    )
                    marking_df["Question #"] = marking_df["Question #"].astype(int)
            
            # If the first question is 1, but our results file starts from 0, adjust
            # This is a heuristic - may need to be adjusted based on specific formats
            if marking_df["Question #"].min() == 1 and marking_df["Question #"].max() == 19:
                logger.info("Adjusting question numbers to start from 0 instead of 1")
                marking_df["Question #"] = marking_df["Question #"] - 1
                    
            logger.info(f"Loaded marking sheet from {marking_sheet_path}")
            return marking_df
            
        except Exception as e:
            logger.error(f"Error loading marking sheet: {e}")
            raise
    
    def calculate_student_scores(self, 
                            results_df: pd.DataFrame, 
                            marking_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate scores for each student based on marking sheet.
        
        Args:
            results_df: DataFrame with student answers
            marking_df: DataFrame with correct answers and points
            
        Returns:
            DataFrame with student scores
        """
        # Create a new dataframe for student scores
        student_scores = pd.DataFrame()
        
        # Get the question number column
        question_num_col = results_df.columns[0]  # This will be "Question" from the example
        
        # Create a dictionary mapping question numbers to correct answers and points
        marking_dict = marking_df.set_index("Question #").to_dict()
        correct_answers = marking_dict.get("Correct Answer Letter", {})
        question_points = marking_dict.get("Points", {})
        
        logger.info(f"Question numbers in results: {sorted(results_df[question_num_col].unique())}")
        logger.info(f"Question numbers in marking sheet: {sorted(marking_df['Question #'].unique())}")
        
        # Process each student column
        for student_col in results_df.columns[1:]:
            # Initialize points for this student
            points = []
            
            # Process each question
            for _, row in results_df.iterrows():
                question_num = row[question_num_col]
                student_answer = row[student_col]
                
                # Convert student answer to uppercase string for comparison
                if pd.notna(student_answer):
                    student_answer = str(student_answer).strip().upper()
                else:
                    student_answer = ""
                    
                # Check if the answer is correct
                correct_answer = correct_answers.get(question_num, "")
                if isinstance(correct_answer, str):
                    correct_answer = correct_answer.strip().upper()
                    
                question_point_value = question_points.get(question_num, 1)
                
                # Award points if correct
                if student_answer and student_answer == correct_answer:
                    points.append(question_point_value)
                else:
                    points.append(0)
                
                # Debug for troubleshooting
                if question_num in [0, 1]:  # Check specific questions
                    logger.debug(f"Q{question_num} - Student: {student_answer}, Correct: {correct_answer}, Points: {points[-1]}")
            
            # Add student's points to the results
            student_scores[f"{student_col}_points"] = points
            # Calculate total score for this student
            student_scores[f"{student_col}_total"] = sum(points)
        
        # Add question numbers and correct answers for reference
        student_scores["Question #"] = results_df[question_num_col]
        student_scores["Correct Answer"] = student_scores["Question #"].map(correct_answers)
        student_scores["Points Possible"] = student_scores["Question #"].map(question_points)
        
        return student_scores
        
    def generate_question_statistics(self, 
                                results_df: pd.DataFrame, 
                                marking_df: pd.DataFrame, 
                                student_scores: pd.DataFrame) -> pd.DataFrame:
        """Generate statistics about each question's performance.
        
        Args:
            results_df: DataFrame with student answers
            marking_df: DataFrame with correct answers and points
            student_scores: DataFrame with calculated student scores
            
        Returns:
            DataFrame with question statistics
        """
        # Get the question number column
        question_num_col = results_df.columns[0]
        
        # Get number of students
        num_students = len(results_df.columns) - 1  # Subtract 1 for the question number column
        
        # Create a dictionary mapping question numbers to correct answers
        marking_dict = marking_df.set_index("Question #").to_dict()
        correct_answers = marking_dict.get("Correct Answer Letter", {})
        question_points = marking_dict.get("Points", {})
        
        # Initialize statistics
        stats = []
        
        # Process each question
        for _, row in results_df.iterrows():
            question_num = row[question_num_col]
            correct_answer = correct_answers.get(question_num, "")
            point_value = question_points.get(question_num, 1)
            
            # Count responses for each possible answer
            response_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'blank': 0}
            
            for student_col in results_df.columns[1:]:
                student_answer = row[student_col]
                if pd.isna(student_answer) or str(student_answer).strip() == "":
                    response_counts['blank'] += 1
                else:
                    answer_key = str(student_answer).strip().upper()
                    if answer_key in response_counts:
                        response_counts[answer_key] += 1
                    else:
                        # Handle invalid responses
                        logger.warning(f"Invalid response '{student_answer}' for question {question_num}")
            
            # Calculate number of correct responses
            num_correct = response_counts.get(correct_answer, 0)
            correct_pct = (num_correct / num_students) * 100 if num_students > 0 else 0
            
            # Calculate discrimination index (point-biserial correlation)
            # This measures how well a question discriminates between high and low performers
            discrimination = self._calculate_discrimination(question_num, student_scores)
            
            # Compile statistics for this question
            stats.append({
                "Question #": question_num,
                "Correct Answer": correct_answer,
                "Points": point_value, 
                "# Correct": num_correct,
                "% Correct": correct_pct,
                "A": response_counts.get('A', 0),
                "B": response_counts.get('B', 0),
                "C": response_counts.get('C', 0),
                "D": response_counts.get('D', 0),
                "Blank": response_counts.get('blank', 0),
                "Discrimination": discrimination
            })
        
        # Create DataFrame from statistics
        stats_df = pd.DataFrame(stats)
        
        return stats_df
    
    def _calculate_discrimination(self, question_num: int, student_scores: pd.DataFrame) -> float:
        """Calculate discrimination index for a question.
        
        Args:
            question_num: Question number
            student_scores: DataFrame with student scores
            
        Returns:
            Discrimination index (-1 to 1, higher means better discrimination)
        """
        # Get rows for this question
        question_data = student_scores[student_scores["Question #"] == question_num]
        
        if len(question_data) == 0:
            return 0
        
        # Get student columns (those containing _points but not _total)
        point_cols = [col for col in student_scores.columns if "_points" in col and "_total" not in col]
        total_cols = [col for col in student_scores.columns if "_total" in col]
        
        if not point_cols or not total_cols:
            return 0
            
        # Get point values for this question for each student
        point_values = []
        total_scores = []
        
        for i, point_col in enumerate(point_cols):
            if i < len(total_cols):
                # Get corresponding total score column
                total_col = total_cols[i]
                
                # Get this question's points for the student
                if not question_data.empty and point_col in question_data.columns:
                    points = question_data[point_col].iloc[0]
                    
                    # Get student's total score excluding this question
                    if not student_scores.empty and total_col in student_scores.columns:
                        total = student_scores[total_col].max()
                        adjusted_total = total - points
                        
                        point_values.append(points)
                        total_scores.append(adjusted_total)
        
        # Calculate correlation coefficient if possible
        if len(point_values) > 1 and len(total_scores) > 1:
            try:
                # Check for zero standard deviation which causes division by zero
                if np.std(point_values) > 0 and np.std(total_scores) > 0:
                    return np.corrcoef(point_values, total_scores)[0, 1]
                else:
                    return 0
            except:
                return 0
        return 0
    
    def generate_summary_report(self, 
                            student_scores: pd.DataFrame, 
                            question_stats: pd.DataFrame, 
                            output_name: str) -> Path:
        """Generate and save summary report with student scores and question statistics.
        
        Args:
            student_scores: DataFrame with student scores
            question_stats: DataFrame with question statistics
            output_name: Base name for output file
            
        Returns:
            Path to saved report
        """
        # Create a summary dataframe for student scores
        student_summary = pd.DataFrame()
        
        # Extract total scores for each student
        total_cols = [col for col in student_scores.columns if "_total" in col]
        
        for col in total_cols:
            student_name = col.replace("_total", "")
            total_score = student_scores[col].max()
            student_summary.at[student_name, "Total Points"] = total_score
        
        # Calculate total possible points
        if "Points Possible" in student_scores.columns:
            total_possible = student_scores["Points Possible"].sum()
            
            # Calculate percentages
            student_summary["Percentage"] = (student_summary["Total Points"] / total_possible) * 100
            
            # Swiss marking system - only report percentages, no letter grades
            logger.info("Using Swiss marking system - reporting only percentages")
        
        # Sort by total points in descending order
        student_summary = student_summary.sort_values("Total Points", ascending=False)
        
        # Create filename
        output_path = self.output_dir / f"{output_name}_summary.xlsx"
        
        # Save to Excel with multiple sheets
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Student summary sheet
            student_summary.to_excel(writer, sheet_name="Student Summary")
            
            # Detailed student scores
            student_scores.to_excel(writer, sheet_name="Detailed Scores", index=False)
            
            # Question statistics
            question_stats.to_excel(writer, sheet_name="Question Statistics", index=False)
            
            # Create class statistics
            class_stats = pd.DataFrame({
                "Statistic": [
                    "Mean", "Median", "Std Dev", "Min", "Max",
                    "# Students", "# Questions", "Perfect Scores (100%)"
                ],
                "Value": [
                    student_summary["Total Points"].mean(),
                    student_summary["Total Points"].median(),
                    student_summary["Total Points"].std(),
                    student_summary["Total Points"].min(),
                    student_summary["Total Points"].max(),
                    len(student_summary),
                    len(question_stats),
                    sum(student_summary["Percentage"] >= 100)
                ]
            })
            
            class_stats.to_excel(writer, sheet_name="Class Statistics", index=False)
        
        # Generate visualization of question difficulty
        self._create_question_difficulty_chart(question_stats, output_name)
        
        # Generate score distribution chart
        self._create_score_distribution_chart(student_summary, output_name)
        
        logger.info(f"Generated summary report: {output_path}")
        
        return output_path
    
    def _create_question_difficulty_chart(self, question_stats: pd.DataFrame, output_name: str) -> Path:
        """Create visualization of question difficulty.
        
        Args:
            question_stats: DataFrame with question statistics
            output_name: Base name for output file
            
        Returns:
            Path to saved chart
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(12, 8))
        
        # Sort questions by question number instead of percent correct
        sorted_stats = question_stats.sort_values("Question #")
        
        # Make sure Question # is treated as a category
        sorted_stats["Question #"] = sorted_stats["Question #"].astype(str)
        
        # Create bar chart - without coloring by question (no hue parameter)
        ax = sns.barplot(x="Question #", y="% Correct", data=sorted_stats, color="steelblue")
        
        # Add reference line at 70%
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='70% Threshold')
        
        # Customize chart
        plt.title("Question Difficulty (% of Students Answering Correctly)")
        plt.xlabel("Question Number")
        plt.ylabel("Percentage Correct")
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        
        # Save chart
        chart_path = self.output_dir / f"{output_name}_question_difficulty.png"
        plt.savefig(chart_path)
        plt.close()
        
        return chart_path
    
    def _create_score_distribution_chart(self, student_summary: pd.DataFrame, output_name: str) -> Path:
        """Create visualization of score distribution.
        
        Args:
            student_summary: DataFrame with student scores
            output_name: Base name for output file
            
        Returns:
            Path to saved chart
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(12, 8))
        
        # Create histogram of scores
        sns.histplot(student_summary["Percentage"], bins=10, kde=True)
        
        # Add reference line at mean value
        mean_pct = student_summary["Percentage"].mean()
        plt.axvline(x=mean_pct, color='red', linestyle='--', alpha=0.7, 
                   label=f"Class Mean ({mean_pct:.1f}%)")
        
        # Add reference line at median value
        median_pct = student_summary["Percentage"].median()
        plt.axvline(x=median_pct, color='green', linestyle='--', alpha=0.7, 
                   label=f"Class Median ({median_pct:.1f}%)")
        
        # Set x-axis limits to always show 0-100% range
        plt.xlim(0, 100)
        
        # Customize chart
        plt.title("Distribution of Student Scores")
        plt.xlabel("Percentage Score")
        plt.ylabel("Number of Students")
        plt.legend()
        plt.tight_layout()
        
        # Save chart
        chart_path = self.output_dir / f"{output_name}_score_distribution.png"
        plt.savefig(chart_path)
        plt.close()
        
        return chart_path

def main():
    """Main entry point for results processor."""
    # Configure argument parser
    parser = argparse.ArgumentParser(description="Process exam results")
    parser.add_argument("results_file", help="Path to Excel file with student answers")
    parser.add_argument("marking_sheet", help="Path to marking sheet Excel file")
    parser.add_argument("--output-dir", default="output", help="Directory to save output files")
    parser.add_argument("--output-name", default="exam_results", help="Base name for output files")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Process results
    processor = ExamResultsProcessor(output_dir=Path(args.output_dir))
    
    # Load data
    results_df = processor.load_results(Path(args.results_file))
    marking_df = processor.load_marking_sheet(Path(args.marking_sheet))
    
    # Calculate scores
    student_scores = processor.calculate_student_scores(results_df, marking_df)
    
    # Generate question statistics
    question_stats = processor.generate_question_statistics(
        results_df, marking_df, student_scores
    )
    
    # Generate and save summary report
    output_path = processor.generate_summary_report(
        student_scores, question_stats, args.output_name
    )
    
    print(f"Results processed successfully. Summary report saved to: {output_path}")

if __name__ == "__main__":
    main()
