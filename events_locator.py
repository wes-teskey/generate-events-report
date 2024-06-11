import os
from textwrap import dedent
import re
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from events_venues import venues_lv
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.tools.tavily_search import TavilySearchResults
import datetime

load_dotenv()

search = TavilySearchAPIWrapper()
search_tool = TavilySearchResults(api_wrapper=search)

@CrewBase
class EventsLocatorCrew():
    agents_config = 'events_agents.yaml'
    tasks_config = 'events_tasks.yaml'
    
    def __init__(self, file_name: str, llm):
        self.file_name = file_name
        self.llm = llm
  
    @agent
    def events_researcher(self) -> Agent:
        return Agent(
          config = self.agents_config['events_researcher'],
          llm = self.llm,
          )
        
    @agent
    def events_writer(self) -> Agent:
        return Agent(
          config = self.agents_config['events_writer'],
          llm = self.llm,
          )
    
    @task
    def events_research_task(self) -> Task:
        return Task(
            config = self.tasks_config['events_research_task'],
            agent = self.events_researcher(),
            tools=[search_tool],
            )
    
    @task
    def events_verification_task(self) -> Task:
        return Task(
            config = self.tasks_config['events_verification_task'],
            agent = self.events_researcher(),
            tools=[search_tool],
            )
    
    @task
    def events_details_research_task(self) -> Task:
        return Task(
            config = self.tasks_config['events_details_research_task'],
            agent = self.events_researcher(),
            tools=[search_tool],
            )
        
    @task
    def events_summary_task(self) -> Task:
        return Task(
            config = self.tasks_config['events_summary_task'],
            agent = self.events_writer(),
            output_file = self.file_name,
            )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents = [self.events_researcher(), self.events_writer()],
            tasks = [self.events_research_task(), self.events_verification_task(),
                     self.events_details_research_task(), self.events_summary_task()],
            process = Process.sequential,
            verbose = 2,
            )
    
def path_to_save_file(sub_directory: str, file_name: str, include_time: bool = True):
    # Path to the current script
    current_script_path = os.path.abspath(__file__)

    # Path two levels up from the current script
    two_levels_up = os.path.dirname(os.path.dirname(current_script_path))

    # Path to the 'data' directory, two levels up from the script
    data_directory = os.path.join(two_levels_up, sub_directory)

    # Specify the full file path for where to write the file
    if include_time:
        # Get the current date and time without milliseconds
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")  # e.g., 2023-04-01_15-30-45
        return os.path.join(data_directory, f'{file_name}_{formatted_time}')
    return os.path.join(data_directory, file_name)

def sanitize_filename(filename):
    # Remove all characters except letters (lowercase and uppercase) and spaces
    sanitized_name = re.sub(r'[^a-zA-Z0-9 _]', '', filename)
    # Replace spaces with underscores after removing leading and trailing spaces
    sanitized_name = sanitized_name.strip()
    sanitized_name = sanitized_name.replace(' ', '_')
    return sanitized_name

def merge_files_with_header(directory, filename_prefix, output_filename, venues):
    # Get list of files in the directory with the specified prefix
    files_to_merge = [f for f in os.listdir(directory) 
                      if f.startswith(filename_prefix)]
    files_to_merge.sort()  # Sort the files in ascending order

    if not files_to_merge:
        print("No files found with the specified prefix.")
        return

    output_filename = path_to_save_file(sub_directory='data',
                                        file_name=output_filename,
                                        include_time=True) + '.txt'
    with open(output_filename, 'w') as outfile:
        for filename in files_to_merge:
            filename_split = filename.split('__')  # Split the filename by '__'
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as infile:
                outfile.write(f"## {venues[int(filename_split[1].split('_')[0]) - 1]} ##\n\n")
                outfile.write(f"{filename_split[1]}\n\n")
                outfile.write(infile.read())
                outfile.write("\n\n\n\n")  # Add a newline for separation between files
            os.remove(filepath)  # Delete the file after merging

def merge_files(venue_loc, dates_to_check, llm_name, venues):
    directory = path_to_save_file(sub_directory='data', file_name= '', include_time=False)
    filename_prefix = sanitize_filename(venue_loc + ' ' + dates_to_check)
    merge_files_with_header(directory, filename_prefix, f'{filename_prefix}_{llm_name}', venues)

def main():
    print("\n\n## Welcome to the Events Finder ##")
    print('-------------------------------')

    dates_to_check = input(
        dedent("""
               What specific dates do you want to check?
               """))
    
    #llm defaults to gpt-4o
    llm_name = "gpt-4o"
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    #Default to Las Vegas Venues
    venue_loc = 'lv' 
    venues = venues_lv

    for i, venue in enumerate(venues):
        output_file_name = path_to_save_file(sub_directory='data',
                                             file_name=sanitize_filename(venue_loc + ' ' +
                                                                         dates_to_check) +
                                                                         '__' + f'{(i + 1):02}' 
                                                                         + '_' +  llm_name,
                                             include_time=True) + '.txt'
        inputs = {'venue': venue, 'dates_to_check': dates_to_check}
        EventsLocatorCrew(output_file_name, llm).crew().kickoff(inputs=inputs)
        
    merge_files(venue_loc, dates_to_check, llm_name, venues)

if __name__ == "__main__":
    main()