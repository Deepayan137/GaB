from src.vqa_data_blip import VQAFineTuneDataset, VQADataset
from Question_type import All_task
import re

from collections import Counter

class AnswerStats:
    def __init__(self, questions, answers):
        self.answers = answers
        self.questions = questions
        self.answer_frequencies = Counter(answers)
        self.question_frequencies = Counter(questions)

    def _num(self, ans=True):
        if not ans:
            return len(self.questions)
        return len(self.answers)

    def _avg_num_words(self, ans=True):
        if not ans:
            words = [len(question.split()) for question in self.questions]
        else:
            words = [len(answer.split()) for answer in self.answers]
        avg_words = sum(words) / len(words)
        return int(avg_words) 

    def _num_unique(self, ans=True):
        if not ans:
            return len(set(question for question in self.questions))
        return len(set(answer for answer in self.answers))

    def _top_frequent(self, n=10, ans=True):
        if not ans:
            return self.question_frequencies.most_common(n)
        return self.answer_frequencies.most_common(n)
    
    def answer_length_distribution(self):
        """Returns a distribution of answer lengths."""
        length_count = Counter(len(answer.split()) for answer in self.answers)
        return dict(length_count)

    def most_common_starting_words(self, n=5, ans=True):
        """Returns the most common starting words of the answers."""
        if not ans:
            starting_words = Counter(' '.join(question.split()[:2]) for question in self.questions if question)
        else:
            starting_words = Counter(answer.split()[0] for answer in self.answers if answer)
        return starting_words.most_common(n)

    def most_common_ending_words(self, n=5, ans=True):
        """Returns the most common ending words of the answers."""
        if not ans:
            li = [' '.join(question.split()[-1]) for question in self.questions if question]
            ending_words = Counter()
        else:
            ending_words = Counter(answer.split()[-1] for answer in self.answers if answer)
        return ending_words.most_common(n)


    def print_stats(self):
        print("Answer Stats")
        print(f"Total answers: {self._num()}")
        print(f"Average number of words per answer: {self._avg_num_words()}")
        print(f"Number of unique answers: {self._num_unique()}")
        print(f"Answer length distribution: {self.answer_length_distribution()}")
        print(f"Most common starting words: {self.most_common_starting_words()}")
        print(f"Most common ending words: {self.most_common_ending_words()}")
        print("Top frequent answers:")
        for i, (answer, frequency) in enumerate(self._top_frequent()):
            print(f"{i}. {answer}: {frequency}")
        print("+++++++++++++++++++++++++++++++")
        print("Question Stats")
        print(f"Average number of words per question: {self._avg_num_words(ans=False)}")
        print(f"Number of unique questions: {self._num_unique(ans=False)}")
        print(f"Most common starting words: {self.most_common_starting_words(ans=False, n=20)}")
        print(f"Most common ending words: {self.most_common_ending_words(ans=False)}")
        print("Top frequent questions:")
        for i, (answer, frequency) in enumerate(self._top_frequent(ans=False)):
            print(f"{i}. {answer}: {frequency}")

if __name__ == "__main__":
    from Question_type import All_task, Category_splits
    from src.param import parse_args
    args = parse_args()
    args.backbone = 'blip'
    task = 'q_subcategory'
    split = 'karpathy_val'
    # train_dset = VQADataset('karpathy_train', True)
    val_dset = VQADataset(split, True)
    # test_dset = VQADataset('karpathy_test', True)
    val_dataset = VQAFineTuneDataset(
        All_task,
        [],
        args=args,
        split=split,
        raw_dataset=val_dset,
        mode=split.split('_')[-1],
        task=task,
        cates=list(i for i in range(80))
        )

    def extract_question(sentence):
        # Regular expression pattern to match the question
        pattern = r'Question: (.*?) Answer:'

        # Search for the pattern in the sentence
        match = re.search(pattern, sentence)

        # Extract and return the question if a match is found
        if match:
            return match.group(1)
        else:
            return "No question found"

    answers = [item['answer'] for item in val_dataset]
    questions = [extract_question(item['sent']) for item in val_dataset]
    print("Calculating Answer Stats")
    stats = AnswerStats(questions, answers)
    stats.print_stats()
