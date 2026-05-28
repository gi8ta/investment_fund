# fund/graph/trading_graph.py

import os
import uuid
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.prebuilt import ToolNode

from fund.agents import *
from fund.default_config import DEFAULT_CONFIG
from fund.agents.utils.memory import FinancialSituationMemory
from fund.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from fund.dataflows.config import set_config

# Import the new abstract tool methods from agent_utils
from fund.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_insider_sentiment,
    get_insider_transactions,
    get_global_news
)

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


class TradingGraph:
    """Main class that orchestrates the multi-agent trading framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
        scope: str = None,
    ):
        """Initialize the trading graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLMs. Explicit timeout/max_retries on every client so a
        # hung HTTP read can't freeze the whole pipeline indefinitely.
        _callbacks = self.config.get("callbacks", [])
        _timeout = self.config.get("llm_request_timeout", 120)
        _retries = self.config.get("llm_max_retries", 2)
        if self.config["llm_provider"].lower() == "openai" or self.config["llm_provider"] == "ollama" or self.config["llm_provider"] == "openrouter":
            self.deep_thinking_llm = ChatOpenAI(model=self.config["deep_think_llm"], base_url=self.config["backend_url"], callbacks=_callbacks, timeout=_timeout, max_retries=_retries)
            self.quick_thinking_llm = ChatOpenAI(model=self.config["quick_think_llm"], base_url=self.config["backend_url"], callbacks=_callbacks, timeout=_timeout, max_retries=_retries)
        elif self.config["llm_provider"].lower() == "anthropic":
            self.deep_thinking_llm = ChatAnthropic(model=self.config["deep_think_llm"], base_url=self.config["backend_url"], callbacks=_callbacks, timeout=_timeout, max_retries=_retries)
            self.quick_thinking_llm = ChatAnthropic(model=self.config["quick_think_llm"], base_url=self.config["backend_url"], callbacks=_callbacks, timeout=_timeout, max_retries=_retries)
        elif self.config["llm_provider"].lower() == "google":
            self.deep_thinking_llm = ChatGoogleGenerativeAI(model=self.config["deep_think_llm"], callbacks=_callbacks, timeout=_timeout, max_retries=_retries)
            self.quick_thinking_llm = ChatGoogleGenerativeAI(model=self.config["quick_think_llm"], callbacks=_callbacks, timeout=_timeout, max_retries=_retries)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config['llm_provider']}")

        # Each TradingGraph instance gets a unique ChromaDB scope so that
        # multiple instances running in the same process (e.g. parallel per-ticker
        # workers) never collide on collection names.
        _scope = scope or uuid.uuid4().hex[:12]

        # Initialize memories
        self.bull_memory         = FinancialSituationMemory(f"bull_{_scope}",         self.config)
        self.bear_memory         = FinancialSituationMemory(f"bear_{_scope}",         self.config)
        self.trader_memory       = FinancialSituationMemory(f"trader_{_scope}",       self.config)
        self.invest_judge_memory = FinancialSituationMemory(f"invest_judge_{_scope}", self.config)
        self.risk_manager_memory = FinancialSituationMemory(f"risk_manager_{_scope}", self.config)

        # Initialize ATLAS-GIC integrations (optional)
        self.darwinian = None
        if self.config.get("enable_darwinian_weights", False):
            from fund.scoring.darwinian import DarwinianWeights
            weights_file = self.config.get("darwinian_weights_file", "darwinian_weights.json")
            self.darwinian = DarwinianWeights(state_file=weights_file)

        self.forward_context_provider = None
        if self.config.get("enable_forward_context", False):
            from fund.context.forward_context import ForwardContextProvider
            self.forward_context_provider = ForwardContextProvider(self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config.get("max_debate_rounds", 1),
            max_risk_discuss_rounds=self.config.get("max_risk_discuss_rounds", 1),
        )
        # Resolve prompts_dir (for autoresearch external prompt loading)
        _prompts_dir = self.config.get("prompts_dir", None)

        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.risk_manager_memory,
            self.conditional_logic,
            darwinian_weights=self.darwinian,
            enable_cro=self.config.get("enable_cro", False),
            cro_rejection_threshold=self.config.get("cro_rejection_threshold", 24),
            forward_context_provider=self.forward_context_provider,
            prompts_dir=_prompts_dir,
            skip_invest_debate=self.config.get("skip_invest_debate", False),
            skip_risk_debate=self.config.get("skip_risk_debate", False),
        )

        # Pass analyst-specific config (lookback windows, etc.)
        self.graph_setup.analyst_config = {
            "market_lookback_days": self.config.get("market_lookback_days", 30),
            "news_lookback_days": self.config.get("news_lookback_days", 7),
        }

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph
        self.graph = self.graph_setup.setup_graph(selected_analysts)

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                ]
            ),
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_sentiment,
                    get_insider_transactions,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # Fundamental analysis tools
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                ]
            ),
        }

    def propagate(self, company_name, trade_date):
        """Run the trading graph for a company on a specific date."""

        self.ticker = company_name

        # Initialize state
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date
        )
        args = self.propagator.get_graph_args()

        if self.debug:
            # Debug mode with tracing
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)

            final_state = trace[-1]
        else:
            # Standard mode without tracing
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection
        self.curr_state = final_state

        # Log state
        self._log_state(trade_date, final_state)

        # Return decision and processed signal
        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        log_entry = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "risky_history": final_state["risk_debate_state"]["risky_history"],
                "safe_history": final_state["risk_debate_state"]["safe_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # Include CRO review if available
        if final_state.get("cro_review"):
            log_entry["cro_review"] = final_state["cro_review"]

        self.log_states_dict[str(trade_date)] = log_entry

        # Save to file
        directory = Path(f"eval_results/{self.ticker}/TradingStrategy_logs/")
        directory.mkdir(parents=True, exist_ok=True)

        with open(
            f"eval_results/{self.ticker}/TradingStrategy_logs/full_states_log_{trade_date}.json",
            "w",
        ) as f:
            json.dump(self.log_states_dict, f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns."""
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_risk_manager(
            self.curr_state, returns_losses, self.risk_manager_memory
        )

    def update_darwinian_weights(self, ticker, trade_date, signal, actual_return_pct):
        """Update Darwinian weights after a trade closes (if enabled)."""
        if self.darwinian and self.curr_state:
            self.darwinian.record_trade_outcome(
                ticker=ticker,
                trade_date=trade_date,
                signal=signal,
                actual_return_pct=actual_return_pct,
                state=self.curr_state,
            )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
