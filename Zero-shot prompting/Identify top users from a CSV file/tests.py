from hstest import dynamic_test, StageTest, CheckResult, TestedProgram
from utils.embeddings_utils import EmbeddingsUtils


class CodeProblemTest(StageTest):

    @dynamic_test(time_limit=60000)
    def test1(self):
        program = TestedProgram()
        output = program.start().strip()

        # Extract the OpenAI API key and GPT response
        lines = output.split('\n')
        openai_api_key = lines[0].split(':', 1)[1].strip()

        # Set the OpenAI API key for EmbeddingsUtils
        EmbeddingsUtils.set_openai_api_key(openai_api_key)

        # Join the lines after the API key for the GPT response
        student_prompt_gpt_response = '\n'.join(lines[1:])

        expected_gpt_response = """```python
import csv

def find_user_with_most_logins(csv_file, month):
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        max_logins = 0
        max_user = None
        for row in reader:
            if row['month'] == month and int(row['number of logins']) > max_logins:
                max_logins = int(row['number of logins'])
                max_user = row['name']
        return max_user
```"""

        # Calculate cosine similarity between the student's response and the expected response
        similarity = CodeProblemTest.get_cosine_similarity(student_prompt_gpt_response, expected_gpt_response)

        # Check if similarity is above the threshold
        if similarity >= 0.85:
            return CheckResult.correct()
        else:
            feedback = "The AI-generated response is not similar enough to the expected output."
            return CheckResult.wrong(feedback)

    @staticmethod
    def get_cosine_similarity(response1, response2):
        embedding1 = EmbeddingsUtils.get_embedding(response1)
        embedding2 = EmbeddingsUtils.get_embedding(response2)
        similarity = 1 - EmbeddingsUtils.distances_from_embeddings(embedding1, [embedding2])[0]
        return similarity


if __name__ == '__main__':
    CodeProblemTest().run_tests()
