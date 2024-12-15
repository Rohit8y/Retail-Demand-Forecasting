class Process:
    def __init__(self, all_data, store_data, test_data, output_file):
        self.all_data = all_data
        self.store_data = store_data
        self.test_data = test_data
        self.output_file = output_file

    def run(self):
        """
        Method to be implemented by subclasses.
        Executes the core processing logic.
        """
        raise NotImplementedError("Subclasses must implement the run() method.")

    def generate_submission(self):
        """
            Method to be implemented by subclasses.
            Executes the core processing logic.
            """
        raise NotImplementedError("Subclasses must implement the generate_submission() method.")
