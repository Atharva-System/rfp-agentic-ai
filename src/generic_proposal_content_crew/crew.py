### File: src/generic_proposal_content_crew/crew.py

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from ..tools import RFPKnowledgeBaseTool
from ..types import OutlineSubsectionContent, SectionAnalysis
import os
import json


@CrewBase
class GenericProposalContentCrew():
    """Crew for dynamically generating proposal content based on user input"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, knowledge_dir: str = "./knowledge", inputs: dict = None):
        self.knowledge_dir = knowledge_dir
        self.inputs = inputs or {}
        self.index_name = self.inputs.get('index_name')
        self.solicitation_id = self.inputs.get('solicitation_id')
        self.processed_files = self.inputs.get('processed_files', [])

        if not self.index_name:
            raise ValueError("OpenSearch index name is required")

        # Create the RFP Knowledge Base Tool (for compatibility with proposal_outline)
        self.rfp_query_tool = RFPKnowledgeBaseTool(
            index_name=self.index_name,
            name="RFP Knowledge Base Tool",
            description="A tool to query the RFP knowledge base for specific requirements, evaluation criteria, and proposal instructions.",
            solicitation_id=self.solicitation_id
        )

        # Set up LLM
        self.llm = LLM(
            model=os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview'),
            api_key=os.getenv('OPENAI_KEY')
        )

    @agent
    def manager_agent(self) -> Agent:
        return Agent(config=self.agents_config['manager_agent'], verbose=True)

    @agent
    def generic_research_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['generic_research_agent'],
            tools=[self.rfp_query_tool],
            verbose=True
        )

    @agent
    def generic_writer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['generic_writer_agent'],
            verbose=True
        )

    @agent
    def technical_research_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_research_agent'],
            tools=[self.rfp_query_tool],
            verbose=True
        )

    @agent
    def technical_writer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_writer_agent'],
            verbose=True
        )

    @agent
    def management_research_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['management_research_agent'],
            tools=[self.rfp_query_tool],
            verbose=True
        )

    @agent
    def management_writer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['management_writer_agent'],
            verbose=True
        )

    @agent
    def resume_research_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['resume_research_agent'],
            tools=[self.rfp_query_tool],
            verbose=True
        )

    @agent
    def resume_writer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['resume_writer_agent'],
            verbose=True
        )

    @agent
    def past_performance_research_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['past_performance_research_agent'],
            tools=[self.rfp_query_tool],
            verbose=True
        )

    @agent
    def past_performance_writer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['past_performance_writer_agent'],
            verbose=True
        )

    @task
    def analyze_section_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_section'],
            output_json=SectionAnalysis
        )

    @task
    def generic_research_task(self) -> Task:
        return Task(config=self.tasks_config['generic_research_task'])

    @task
    def generic_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config['generic_writing_task'],
            output_json=OutlineSubsectionContent,
        )

    @task
    def technical_research_task(self) -> Task:
        return Task(config=self.tasks_config['technical_research_task'])

    @task
    def technical_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config['technical_writing_task'],
            output_json=OutlineSubsectionContent,
        )

    @task
    def management_research_task(self) -> Task:
        return Task(config=self.tasks_config['management_research_task'])

    @task
    def management_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config['management_writing_task'],
            output_json=OutlineSubsectionContent,
        )

    @task
    def resume_research_task(self) -> Task:
        return Task(config=self.tasks_config['resume_research_task'])

    @task
    def resume_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config['resume_writing_task'],
            output_json=OutlineSubsectionContent,
        )

    @task
    def past_performance_research_task(self) -> Task:
        return Task(config=self.tasks_config['past_performance_research_task'])

    @task
    def past_performance_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config['past_performance_writing_task'],
            output_json=OutlineSubsectionContent,
        )

    def _get_task_sequence(self, analysis: SectionAnalysis):
        """Determine task sequence based on manager's analysis"""
        if analysis.task_type == 'technical':
            return [self.technical_research_task(), self.technical_writing_task()] if analysis.is_research else [self.technical_writing_task()]
        elif analysis.task_type == 'management':
            return [self.management_research_task(), self.management_writing_task()] if analysis.is_research else [self.management_writing_task()]
        elif analysis.task_type == 'resume':
            return [self.resume_research_task(), self.resume_writing_task()] if analysis.is_research else [self.resume_writing_task()]
        elif analysis.task_type == 'past_performance':
            return [self.past_performance_research_task(), self.past_performance_writing_task()] if analysis.is_research else [self.past_performance_writing_task()]
        else:
            return [self.generic_research_task(), self.generic_writing_task()] if analysis.is_research else [self.generic_writing_task()]

    def _parse_analysis_output(self, analysis_result):
        """Parse the analysis output from the crew result"""
        try:
            task_output = analysis_result.tasks_output[0]
            if isinstance(task_output, dict):
                return SectionAnalysis(**task_output)
            if isinstance(task_output, str):
                try:
                    data = json.loads(task_output)
                    return SectionAnalysis(**data)
                except json.JSONDecodeError:
                    task_type = 'generic'
                    is_research = True
                    reasoning = task_output
                    if 'technical' in task_output.lower():
                        task_type = 'technical'
                    elif 'management' in task_output.lower():
                        task_type = 'management'
                    elif 'resume' in task_output.lower():
                        task_type = 'resume'
                    elif 'past performance' in task_output.lower():
                        task_type = 'past_performance'
                    is_research = 'research' in task_output.lower()
                    return SectionAnalysis(
                        task_type=task_type,
                        is_research=is_research,
                        reasoning=reasoning
                    )
            return SectionAnalysis(
                task_type='generic',
                is_research=True,
                reasoning="Could not parse analysis output, defaulting to generic task"
            )
        except Exception as e:
            return SectionAnalysis(
                task_type='generic',
                is_research=True,
                reasoning=f"Error parsing analysis: {str(e)}, defaulting to generic task"
            )

    @crew
    def crew(self) -> Crew:
        analysis_crew = Crew(
            agents=[self.manager_agent()],
            tasks=[self.analyze_section_task()],
            process=Process.sequential,
            verbose=True,
            LLM=self.llm
        )
        analysis_result = analysis_crew.kickoff()
        analysis = self._parse_analysis_output(analysis_result)
        tasks = self._get_task_sequence(analysis)
        # Dynamically select agents based on analysis.task_type
        agents = [self.manager_agent()]
        if analysis.task_type == 'technical':
            agents.extend([self.technical_research_agent(), self.technical_writer_agent()])
        elif analysis.task_type == 'management':
            agents.extend([self.management_research_agent(), self.management_writer_agent()])
        elif analysis.task_type == 'resume':
            agents.extend([self.resume_research_agent(), self.resume_writer_agent()])
        elif analysis.task_type == 'past_performance':
            agents.extend([self.past_performance_research_agent(), self.past_performance_writer_agent()])
        else:
            agents.extend([self.generic_research_agent(), self.generic_writer_agent()])
        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            LLM=self.llm
        )

# Kickoff function
async def kickoff_generic_proposal_content_crew(knowledge_dir: str = "./knowledge", inputs: dict = {}):
    crew_instance = GenericProposalContentCrew(knowledge_dir=knowledge_dir, inputs=inputs)
    result = await crew_instance.crew().kickoff_async(inputs=inputs)
    return result
