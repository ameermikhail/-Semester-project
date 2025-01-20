# -Semester-project
## Overview
The Salon Appointment Scheduler is an advanced software application tailored for managing and optimizing appointments within a salon environment. 
It features a user-friendly graphical interface and leverages a Genetic Algorithm (GA) to enhance the scheduling process. 
The primary purpose of this GA is to minimize scheduling conflicts and maximize resource utilization, ultimately improving both customer satisfaction and business performance.

### Purpose of the Genetic Algorithm (GA)
The GA is central to our scheduling system, designed to simulate evolutionary processes to produce highly efficient schedules.
Unlike traditional scheduling methods, which might rely on static rules or simple optimization routines, the GA iteratively improves the quality of the schedule by mimicking natural selection.

Here's how it defines and seeks a better result:
- **Idle Time Reduction**: The GA evaluates each potential schedule by calculating the total idle time between appointments, A lower idle time indicates a more compact schedule, allowing staff to maximize their productivity.
 Reducing idle time is crucial for keeping operational costs in check while ensuring employees are effectively utilized.

- **Revenue Maximization**: Each schedule is also assessed based on the potential revenue it can generate, This involves not only filling more appointment slots but also strategically scheduling high-value services at times that maximize customer attendance.
 The GA prioritizes schedules that balance high utilization with optimal service placement to boost overall earnings.

### Possible Changes by the GA
The GA dynamically adapts schedules based on current booking trends, staff availability, and service demand.

Here are some possible changes it could make:
- **Schedule Rearrangement**: The GA can shift appointments to different times or dates to better align with staff shifts or to consolidate appointments, reducing gaps in the schedule.

- **Service Prioritization**: High-demand or higher-priced services can be prioritized during peak times, or spread throughout the day to ensure consistent revenue flow.

- **Staff Allocation**: By analyzing patterns in service demand and staff skills, the GA may reallocate certain services to different stylists to optimize their workload and expertise utilization.

The GA operates under a set framework of evolutionary operations, including selection (choosing the best schedules based on a fitness function that considers both idle time and revenue), crossover (combining features of two schedules), and mutation (introducing random changes to prevent local maxima and encourage diverse solutions). Through these mechanisms, the GA ensures that the scheduling system continuously evolves and adapts, yielding progressively better results without the need for explicit training on historical data. This makes it ideally suited for the dynamic, varied demands of salon scheduling.

By implementing these strategies, the Salon Appointment Scheduler not only meets the immediate needs of its users but also adapts to changing conditions, ensuring long-term efficiency and profitability.

## Features

- **Database Management**: Utilizes SQLite to manage and store customer and appointment data.
- **Text-to-Speech**: Provides audible interaction, improving user accessibility by vocalizing responses.
- **Speech Recognition**: Facilitates speech-to-text capabilities for input, using the Vosk API to recognize and process spoken commands.
- **Dynamic Scheduling**:
  - **Greedy Algorithm**: Allocates appointments as they come, focusing on immediate availability.
  - **Genetic Algorithm (GA)**: Simulates evolutionary processes to find the most efficient scheduling over time without the need for traditional "training". This algorithm works as follows:
    - **Initial Population**: Generates random schedules serving as the starting population.
    - **Fitness Function**: Evaluates each schedule based on the number of scheduling conflicts, total idle time between appointments, and potential revenue.
    - **Evolutionary Operations**: Includes selection (choosing the best schedules), crossover (combining features of two schedules), and mutation (randomly altering part of a schedule).
    - **Optimization**: Iteratively refines schedules, aiming to enhance overall scheduling efficiency across generations.
    - **No Need for Training**: Unlike machine learning models, GA does not require training on historical data. It generates solutions based on defined fitness criteria and uses evolutionary strategies to optimize these solutions, making it ideal for dynamic environments like salon scheduling where conditions and preferences may frequently change.
- **Graphical User Interface**: Designed with customtkinter, providing a clean and modern user experience.
- **Comprehensive Management Tools**: Facilitates adding, viewing, and searching for appointments and customer data, supported by a robust backend.
- **AI-Powered Interactions**: Engages users with an AI conversational agent for assistance, enhancing user interaction.
- **Analytics Dashboard**: Offers visual insights into business metrics like appointment volume, popular services, and revenue.

## Genetic Algorithm Main Function and Parameters

The Genetic Algorithm is utilized to optimize the scheduling of appointments. The main function, run_genetic_algorithm, orchestrates this optimization using the following parameters:

- **Generations (300)**: Specifies the number of iterations the GA will run, allowing the solution pool to evolve over time.
- **Population Size (100)**: Defines the number of potential solutions (schedules) generated initially and maintained throughout the algorithm's run.
- **Mutation Rate (0.1)**: The probability with which random changes are introduced to solutions, helping to explore new areas of the solution space and maintain genetic diversity.
- **Elitism Size (10)**: The number of top solutions that are automatically carried over to the next generation, ensuring that successful traits are preserved.

Each parameter plays a critical role:
- **Generations**: More generations allow for more thorough exploration but increase computational demand.
- **Population Size**: A larger population offers a broader genetic base and improves the likelihood of a strong solution but requires more processing power.
- **Mutation Rate**: Higher rates increase diversity but can destabilize convergence to an optimal solution.
- **Elitism Size**: Ensuring that top performers are preserved prevents loss of good traits but can lead to premature convergence if set too high.

## Installation

1. **Install Dependencies**:
   Ensure Python is installed on your system, then install the following packages using pip. These libraries are necessary for running the application:
   - customtkinter: Used for creating a modern graphical user interface.
   - tkcalendar: Provides calendar widgets for tkinter.
   - sqlite3: Handles local database storage.
   - pyttsx3: Text-to-speech conversion library.
   - matplotlib: For generating statistical graphics.
   - json: For parsing and outputting JSON.
   - pyaudio: Interface for capturing and playing audio on a variety of platforms.
   - vosk: Speech recognition toolkit.
   - dateparser: For parsing dates from natural language input.
   - word2number: Converts number words (e.g., "two") into numeric form.
   
bash
   pip install customtkinter tkcalendar sqlite3 pyttsx3 matplotlib json pyaudio vosk dateparser word2number


2. **Setup Speech Recognition with Vosk**:
   - **Model Download**: Visit [Vosk API Models](https://alphacephei.com/vosk/models) to download the appropriate Vosk model for your language. For English, the vosk-model-small-en-us-0.15 or vosk-model-en-us-0.22 are recommended for their balance of performance and size.
   - **Model Installation**: After downloading, unzip the model to a directory accessible by your script. For example, if you're running your script from the directory C:/projects/salon_scheduler/, you might unzip your model to C:/projects/salon_scheduler/vosk-model.
   - **Configuration**: In your script, set the model_path to the path where you've stored the Vosk model files. For example:
     
python
     model_path = 'C:/projects/salon_scheduler/vosk-model'

   - **Usage**: Ensure that your application correctly initializes Vosk with this model path when setting up speech recognition functionalities.

## Running the Application

- **Script Location**: Place the script in any directory of your choice and navigate to that directory using a terminal or command prompt.
- **Execute the script** to start the application:
   
bash
   python main.py

- **Navigate the GUI** to manage appointments and interact with the system functionalities.

## Project Structure

- **Main Application**: Contains the primary business logic and the GUI.
- **Database Management**: Manages connections, queries, and updates to the SQLite database.
- **Scheduling Algorithms**: Includes implementations of both the Greedy and Genetic algorithms for appointment scheduling.
- **Speech and Text Handling**: Handles speech recognition and text-to-speech output for user interaction.
- **Analytics and Reporting**: Provides tools and functions to generate reports and visualizations of business data.

## Comparisons and Output

- **Comparison Output**: The comparison between the Greedy and Genetic scheduling algorithms' results is displayed on the terminal. This output includes metrics such as the number of conflicts, idle times, and total revenue, facilitating a direct comparison of their efficiency. This comparison is not part of the GUI but is designed to be viewed in the terminal for detailed analysis.

## Note
A ready dataset will be provided for initial setup to facilitate a quick start with pre-populated data.
Three screenshots demonstrating the application's runtime results have been attached to showcase the functionality and output of the system.
