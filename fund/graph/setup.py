# fund/graph/setup.py

from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from fund.agents import *
from fund.agents.utils.agent_states import AgentState

from .conditional_logic import ConditionalLogic


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: ChatOpenAI,
        deep_thinking_llm: ChatOpenAI,
        tool_nodes: Dict[str, ToolNode],
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        risk_manager_memory,
        conditional_logic: ConditionalLogic,
        darwinian_weights=None,
        enable_cro: bool = False,
        cro_rejection_threshold: int = 24,
        forward_context_provider=None,
        prompts_dir=None,
        skip_invest_debate: bool = False,
        skip_risk_debate: bool = False,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.bull_memory = bull_memory
        self.bear_memory = bear_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.risk_manager_memory = risk_manager_memory
        self.conditional_logic = conditional_logic
        self.darwinian_weights = darwinian_weights
        self.enable_cro = enable_cro
        self.cro_rejection_threshold = cro_rejection_threshold
        self.forward_context_provider = forward_context_provider
        self.prompts_dir = prompts_dir
        self.skip_invest_debate = skip_invest_debate
        self.skip_risk_debate = skip_risk_debate
        self.analyst_config = None  # Set by TradingGraph

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals"]
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Graph Setup Error: no analysts selected!")

        # Create analyst nodes (with optional forward context and config)
        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}
        fwd = self.forward_context_provider
        acfg = self.analyst_config

        if "market" in selected_analysts:
            analyst_nodes["market"] = create_market_analyst(
                self.quick_thinking_llm, forward_context_provider=fwd, config=acfg
            )
            delete_nodes["market"] = create_msg_delete()
            tool_nodes["market"] = self.tool_nodes["market"]

        if "social" in selected_analysts:
            analyst_nodes["social"] = create_social_media_analyst(
                self.quick_thinking_llm, forward_context_provider=fwd, config=acfg
            )
            delete_nodes["social"] = create_msg_delete()
            tool_nodes["social"] = self.tool_nodes["social"]

        if "news" in selected_analysts:
            analyst_nodes["news"] = create_news_analyst(
                self.quick_thinking_llm, forward_context_provider=fwd, config=acfg
            )
            delete_nodes["news"] = create_msg_delete()
            tool_nodes["news"] = self.tool_nodes["news"]

        if "fundamentals" in selected_analysts:
            analyst_nodes["fundamentals"] = create_fundamentals_analyst(
                self.quick_thinking_llm, forward_context_provider=fwd, config=acfg
            )
            delete_nodes["fundamentals"] = create_msg_delete()
            tool_nodes["fundamentals"] = self.tool_nodes["fundamentals"]

        pd = self.prompts_dir

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(
            self.quick_thinking_llm, self.bull_memory, prompts_dir=pd
        )
        bear_researcher_node = create_bear_researcher(
            self.quick_thinking_llm, self.bear_memory, prompts_dir=pd
        )
        research_manager_node = create_research_manager(
            self.deep_thinking_llm, self.invest_judge_memory,
            darwinian_weights=self.darwinian_weights, prompts_dir=pd,
        )
        trader_node = create_trader(
            self.quick_thinking_llm, self.trader_memory, prompts_dir=pd
        )

        # Create risk analysis nodes
        risky_analyst = create_risky_debator(self.quick_thinking_llm, prompts_dir=pd)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm, prompts_dir=pd)
        safe_analyst = create_safe_debator(self.quick_thinking_llm, prompts_dir=pd)
        risk_manager_node = create_risk_manager(
            self.deep_thinking_llm, self.risk_manager_memory,
            darwinian_weights=self.darwinian_weights, prompts_dir=pd,
        )

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add analyst nodes to the graph
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(
                f"Msg Clear {analyst_type.capitalize()}", delete_nodes[analyst_type]
            )
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        # ── Add nodes based on ablation mode ─────────────────────

        # Investment debate nodes (skippable for ablation)
        if not self.skip_invest_debate:
            workflow.add_node("Bull Researcher", bull_researcher_node)
            workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)

        # Risk debate nodes (skippable for ablation)
        if not self.skip_risk_debate:
            workflow.add_node("Risky Analyst", risky_analyst)
            workflow.add_node("Neutral Analyst", neutral_analyst)
            workflow.add_node("Safe Analyst", safe_analyst)
        workflow.add_node("Risk Judge", risk_manager_node)

        # Optionally add CRO Reviewer node
        if self.enable_cro:
            from fund.agents.risk_mgmt.cro_reviewer import create_cro_reviewer
            cro_reviewer_node = create_cro_reviewer(
                self.deep_thinking_llm, prompts_dir=pd,
                rejection_threshold=self.cro_rejection_threshold,
            )
            workflow.add_node("CRO Reviewer", cro_reviewer_node)

        # ── Define edges ─────────────────────────────────────────

        # Start with the first analyst
        first_analyst = selected_analysts[0]
        workflow.add_edge(START, f"{first_analyst.capitalize()} Analyst")

        # Connect analysts in sequence
        for i, analyst_type in enumerate(selected_analysts):
            current_analyst = f"{analyst_type.capitalize()} Analyst"
            current_tools = f"tools_{analyst_type}"
            current_clear = f"Msg Clear {analyst_type.capitalize()}"

            # Add conditional edges for current analyst
            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{analyst_type}"),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            # Connect to next analyst or downstream
            if i < len(selected_analysts) - 1:
                next_analyst = f"{selected_analysts[i+1].capitalize()} Analyst"
                workflow.add_edge(current_clear, next_analyst)
            else:
                # Last analyst → either Bull Researcher (normal) or Research Manager (ablation)
                if self.skip_invest_debate:
                    workflow.add_edge(current_clear, "Research Manager")
                else:
                    workflow.add_edge(current_clear, "Bull Researcher")

        # Investment debate edges (only if not skipped)
        if not self.skip_invest_debate:
            workflow.add_conditional_edges(
                "Bull Researcher",
                self.conditional_logic.should_continue_debate,
                {
                    "Bear Researcher": "Bear Researcher",
                    "Research Manager": "Research Manager",
                },
            )
            workflow.add_conditional_edges(
                "Bear Researcher",
                self.conditional_logic.should_continue_debate,
                {
                    "Bull Researcher": "Bull Researcher",
                    "Research Manager": "Research Manager",
                },
            )

        # Research Manager → Trader
        workflow.add_edge("Research Manager", "Trader")

        # Trader → either Risky Analyst (normal) or Risk Judge (ablation)
        if self.skip_risk_debate:
            workflow.add_edge("Trader", "Risk Judge")
        else:
            workflow.add_edge("Trader", "Risky Analyst")
            workflow.add_conditional_edges(
                "Risky Analyst",
                self.conditional_logic.should_continue_risk_analysis,
                {
                    "Safe Analyst": "Safe Analyst",
                    "Risk Judge": "Risk Judge",
                },
            )
            workflow.add_conditional_edges(
                "Safe Analyst",
                self.conditional_logic.should_continue_risk_analysis,
                {
                    "Neutral Analyst": "Neutral Analyst",
                    "Risk Judge": "Risk Judge",
                },
            )
            workflow.add_conditional_edges(
                "Neutral Analyst",
                self.conditional_logic.should_continue_risk_analysis,
                {
                    "Risky Analyst": "Risky Analyst",
                    "Risk Judge": "Risk Judge",
                },
            )

        # Final edge: Risk Judge -> CRO (if enabled) -> END
        if self.enable_cro:
            workflow.add_edge("Risk Judge", "CRO Reviewer")
            workflow.add_edge("CRO Reviewer", END)
        else:
            workflow.add_edge("Risk Judge", END)

        # Compile and return
        return workflow.compile()
