"""Newsletter Automata: A graph-based workflow for generating AI newsletters from Discord content."""

import os
import sys
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import asyncio
from pydantic import Field, BaseModel
import traceback

# Add parent directory to path
file = os.path.abspath(__file__)
parent = os.path.dirname(os.path.dirname(os.path.dirname(file)))
sys.path.insert(0, parent)

from babelgraph.core.graph import Graph, NodeState, NodeContext, NodeStatus
from babelgraph.core.nodes.base import LLMNode
from babelgraph.core.nodes.decisions import BinaryDecisionNode
from babelgraph.core.agent import BaseAgent
from babelgraph.core.prompts.persona import KEN_E
from babelgraph.tools import DiscordTools
from babelgraph.core.runtime import RuntimeConfig
from babelgraph.core.state import NodeState as NewNodeState

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONSTANTS
PERSONA = KEN_E
MESSAGES_TO_FETCH = 1  # For testing, we'll fetch just one message
LOOKBACK_DAYS = None  # Disable lookback days for now

# Load channel configuration
def load_channel_config() -> tuple[Dict[str, str], Dict[str, List[str]]]:
    """Load channel configuration from config/channels.json."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                              'config', 'channels.json')
    try:
        with open(config_path) as f:
            config = json.load(f)
            return config["channels"], config["categories"]
    except Exception as e:
        logger.error(f"Error loading channel configuration: {str(e)}")
        raise

# Channel Configuration
CHANNELS, CATEGORIES = load_channel_config()

TARGET_AUDIENCE =   """
                    Technologists. Entrepreneurs. Developers. Creators. Founders. Business leaders. 
                    All interested in going deep on AI to enhance their work and business.
                    """       

# Load example newsletter template
def load_example_newsletter() -> str:
    """Load the example newsletter template."""
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'example_newsletter.md')
    try:
        with open(template_path) as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading example newsletter: {str(e)}")
        return ""

EXAMPLE_NEWSLETTER = load_example_newsletter()

# Newsletter Prompt Templates
NEWSLETTER_IDENTITY = f"""
Newsletter Identity:
- Title: "Improbable Automata: Field notes from the AI frontier"
- Subtitle: "(hallucinations may vary)"
- Target Audience: {TARGET_AUDIENCE}

Example Newsletter Format:
{EXAMPLE_NEWSLETTER}
"""

NEWSLETTER_STRUCTURE = """
Structure:
1. Synopsis
    - 1-3 passages summarizing the most interesting and relevant content from the week
    - Written in plain but engaging language
    - Draw excitement with compelling hooks
    - Free-flowing, natural style, keeping it interesting for readers
    - Embed links to best content

2. <custom title>
    - The first section after field notes is a curated list of the best content from the week, title is custom to the vibe of the week
    - We care about business relevance, technical insights, and practical applications
    - Flow is more structured for this section. use a mix of bullet points and short paragraphs to keep it interesting
    - We want to include relevant links in each section while maintaining flow

3. Closing Thoughts
   - Brief, impactful final words
   - Connect to broader themes
"""

WRITING_STYLE = """
Writing Style:
- Informative yet playful
- Technical concepts through clear metaphors
- Assume intelligence, explain complexity
- Balance wonder with practicality
- Use humor to illuminate, not distract

Voice:
- Knowledgeable but approachable
- Excited about possibilities
- Grounded in practical reality
- Speaking peer-to-peer with tech leaders
- Focus on business value and implementation
"""

ANALYSIS_PROMPT = f"""Analyze these Discord messages for Improbable Automata newsletter:
Messages: {{collect[content]}}
Date Range: {{collect[date_range]}}

Target Audience: {TARGET_AUDIENCE}

Consider:
1. Who does this content help?
2. Is this content highly technical or academic?
3. Does this content need practical explanation?
4. Are there ways we can think of making use of this technology or new development to improve or work or products?
5. Does this content point to any trends or interesting ideas?

For each shared content piece, consider the above and note your thoughts. 
Keep in mind the desire to make this content useful and actionable for the target audience.
Keep it technically interesting, but practical and insightful.
"""

DRAFT_PROMPT = f"""Generate an engaging field notes entry for Improbable Automata:

Analysis: {{analyze[response]}}

{NEWSLETTER_IDENTITY}
{NEWSLETTER_STRUCTURE}

Follow the style and formatting of the example newsletter above, including:
1. Proper markdown formatting with headers, emphasis, and links
2. Clear section organization
3. Engaging writing style that balances technical depth with accessibility
4. Proper citation and linking of sources
5. Professional yet conversational tone

Your newsletter should maintain similar quality and style while covering the current content."""

class ContentCollectorNode(LLMNode):
    """Node that collects and processes Discord messages from multiple channels."""
    
    channels: Dict[str, str] = Field(default_factory=dict)
    categories: Dict[str, List[str]] = Field(default_factory=dict)
    lookback_days: Optional[int] = Field(default=LOOKBACK_DAYS)
    parallel: bool = True  # Enable parallel processing
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process Discord messages into structured content."""
        try:
            # Initialize Discord toolkit
            toolkit = DiscordTools(
                channels=self.channels,
                categories=self.categories
            )
            
            # Calculate lookback time if specified
            after = None
            if self.lookback_days:
                after = datetime.now() - timedelta(days=self.lookback_days)
                logger.info(f"📅 Looking back to: {after.isoformat()}")
            
            # Process channels in parallel
            target_channels = ["ai-news", "content-stream"]
            tasks = []
            for channel_name in target_channels:
                tasks.append(self._process_channel(toolkit, channel_name, after))
            
            # Wait for all channel processing to complete
            results = await asyncio.gather(*tasks)
            
            # Combine results
            all_messages = {}
            processed_content = []
            link_collection = []
            total_messages = 0
            
            for channel_result in results:
                channel_name = channel_result["channel"]
                all_messages[channel_name] = channel_result["messages"]
                processed_content.extend(channel_result["processed"])
                link_collection.extend(channel_result["links"])
                total_messages += len(channel_result["messages"])
            
            # Store results in state
            state.results[self.id] = {
                "content": processed_content,
                "total_messages": total_messages,
                "channel_stats": {
                    channel: len(msgs) 
                    for channel, msgs in all_messages.items()
                },
                "date_range": f"{after.isoformat() if after else 'N/A'} to {datetime.now().isoformat()}",
                "link_collection": link_collection,
                "response": f"Collected {total_messages} messages with {len(link_collection)} links from {len(all_messages)} channels"
            }
            
            # Log collection summary
            logger.info("\n📊 Content Collection Summary:")
            for channel, count in state.results[self.id]["channel_stats"].items():
                logger.info(f"  #{channel}: {count} messages")
            logger.info(f"  🔗 Total links: {len(link_collection)}")
            
            state.mark_status(self.id, NodeStatus.COMPLETED)
            return "analyze"
            
        except Exception as e:
            logger.error(f"Error collecting content: {str(e)}")
            state.results[self.id] = {"error": str(e)}
            state.mark_status(self.id, NodeStatus.ERROR)
            return None

    async def _process_channel(self, toolkit: DiscordTools, channel_name: str, after: Optional[datetime]) -> Dict[str, Any]:
        """Process a single channel and return its results."""
        try:
            logger.info(f"📨 Reading from #{channel_name}")
            messages = await toolkit.read_channel(
                channel_name=channel_name,
                after=after,
                limit=MESSAGES_TO_FETCH
            )
            logger.info(f"✅ Retrieved {len(messages)} messages from #{channel_name}")
            
            # Debug log to see raw message structure
            for i, msg in enumerate(messages):
                logger.info(f"Raw message {i} from #{channel_name}: {msg}")
            
            # Process messages from this channel
            processed = []
            links = []
            for msg in messages:
                try:
                    processed_msg = self._process_message(msg, channel_name)
                    processed.append(processed_msg)
                    links.extend(processed_msg["links"])
                except Exception as msg_error:
                    logger.error(f"Error processing message in #{channel_name}: {str(msg_error)}")
                    logger.error(f"Message that caused error: {msg}")
                    continue
            
            return {
                "channel": channel_name,
                "messages": messages,
                "processed": processed,
                "links": links
            }
            
        except Exception as e:
            logger.error(f"❌ Error reading #{channel_name}: {str(e)}")
            return {
                "channel": channel_name,
                "messages": [],
                "processed": [],
                "links": []
            }

    def _process_message(self, msg: Dict[str, Any], channel: str) -> Dict[str, Any]:
        """Process a single message into a standardized format."""
        try:
            # Handle string messages (convert to dict)
            if isinstance(msg, str):
                msg = {
                    "content": msg,
                    "timestamp": datetime.now().isoformat(),
                    "author": "unknown",
                    "embeds": [],
                    "attachments": []
                }
            
            content_links = []
            
            # Extract URLs from content if present
            content = msg.get("content", "")
            if isinstance(content, str) and "http" in content:
                # Simple URL extraction - could be enhanced with regex
                for word in content.split():
                    if word.startswith(("http://", "https://")):
                        # Keep full context for model processing
                        link_data = {
                            "url": word,
                            "context": content,
                            "timestamp": msg.get("timestamp"),
                            "source": f"direct_share_{channel}"
                        }
                        content_links.append(link_data)
                        # Log truncated version
                        log_context = content[:100] + "..." if len(content) > 100 else content
                        logger.info(f"Found link: {word} (context: {log_context})")
            
            # Process embeds
            embeds = msg.get("embeds", [])
            if isinstance(embeds, list):
                for embed in embeds:
                    if not isinstance(embed, dict):
                        logger.warning(f"Skipping invalid embed format: {type(embed)}")
                        continue
                        
                    try:
                        # Extract links from description
                        description = embed.get("description", "")
                        if isinstance(description, str):
                            # Look for markdown links [text](url)
                            import re
                            markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', description)
                            for text, url in markdown_links:
                                # Keep full context for model processing
                                link_data = {
                                    "url": url,
                                    "title": text,
                                    "context": description,
                                    "timestamp": msg.get("timestamp"),
                                    "source": f"embed_{channel}"
                                }
                                content_links.append(link_data)
                                # Log truncated version
                                log_desc = description[:100] + "..." if len(description) > 100 else description
                                logger.info(f"Found markdown link: {text} - {url} (context: {log_desc})")
                        
                        # Get embed URL if present
                        url = embed.get("url")
                        if url:
                            # Handle image field which can be string or dict
                            image = embed.get("image")
                            image_url = image.get("url") if isinstance(image, dict) else image
                            
                            # Keep full context for model processing
                            link_data = {
                                "url": url,
                                "title": embed.get("title"),
                                "description": embed.get("description"),
                                "timestamp": msg.get("timestamp"),
                                "source": f"embed_{channel}",
                                "image": image_url
                            }
                            content_links.append(link_data)
                            # Log truncated version
                            log_desc = str(embed.get("description"))[:100] + "..." if embed.get("description") and len(str(embed.get("description"))) > 100 else embed.get("description")
                            logger.info(f"Found embed URL: {url} (description: {log_desc})")
                    except Exception as embed_error:
                        logger.error(f"Error processing embed: {str(embed_error)}")
                        logger.error(f"Problematic embed: {embed}")
                        continue
            
            # Create processed message
            return {
                "channel": channel,
                "id": msg.get("id"),
                "content": content,
                "timestamp": msg.get("timestamp"),
                "author": msg.get("author"),
                "embeds": embeds,
                "attachments": msg.get("attachments", []),
                "links": content_links
            }
            
        except Exception as e:
            logger.error(f"Error in _process_message: {str(e)}")
            logger.error(f"Message that caused error: {msg}")
            raise

class NewsletterFormatterNode(LLMNode):
    """Node that formats content according to Improbable Automata style guide."""
    
    # Declare additional_context as a proper field
    additional_context: str = Field(default="")
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Format the newsletter draft according to style guidelines."""
        try:
            draft = state.results.get("enrich", {}).get("response")
            if not draft:
                logger.warning("No enriched content to format")
                state.mark_status(self.id, NodeStatus.ERROR)
                return None
            
            # Create complete newsletter template with context
            newsletter_template = create_newsletter_prompt(self.additional_context)
            
            # Format according to style guide using LLMNode's process
            formatted_prompt = FORMAT_PROMPT.format(
                draft=draft,
                newsletter_template=newsletter_template
            )
            self.prompt = formatted_prompt
            
            # Use parent class process method which handles LLM interaction
            result = await super().process(state)
            
            if result:
                state.mark_status(self.id, NodeStatus.COMPLETED)
                return None  # Terminal node
            
            state.mark_status(self.id, NodeStatus.ERROR)
            return None
            
        except Exception as e:
            logger.error(f"Error formatting newsletter: {str(e)}")
            state.results[self.id] = {"error": str(e)}
            state.mark_status(self.id, NodeStatus.ERROR)
            return None

def save_newsletter_to_file(content: str, date: datetime) -> str:
    """Save newsletter content to a markdown file.
    
    Args:
        content: The newsletter content
        date: The date of the newsletter
        
    Returns:
        str: Path to the saved file
    """
    # Create generations directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), "generations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = date.strftime('%Y%m%d_%H%M%S')
    filename = f"newsletter_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)
    
    logger.info(f"💾 Saving newsletter to: {filepath}")
    
    # Write content
    with open(filepath, "w") as f:
        f.write(content)
    
    logger.info(f"✅ Newsletter saved successfully ({os.path.getsize(filepath)} bytes)")
    return filepath

class LinkEnricherNode(LLMNode):
    """Node that enriches content with additional context and metadata."""
    
    async def process(self, state: NodeState) -> Optional[str]:
        """Process and enrich the draft content."""
        try:
            draft = state.results.get("draft", {}).get("response")
            if not draft:
                logger.warning("No draft content to enrich")
                state.mark_status(self.id, NodeStatus.ERROR)
                return None
            
            # Extract all links from the content
            links = state.results.get("collect", {}).get("link_collection", [])
            
            # Add metadata about sources
            metadata = {
                "total_sources": len(links),
                "channels": state.results["collect"]["channel_stats"],
                "date_range": state.results["collect"]["date_range"]
            }
            
            # Add metadata header to draft
            enriched = f"""---
Date Range: {metadata['date_range']}
Sources: {metadata['total_sources']} links from {len(metadata['channels'])} channels
---

{draft}"""
            
            state.results[self.id] = {"response": enriched}
            state.mark_status(self.id, NodeStatus.COMPLETED)
            return "format"
            
        except Exception as e:
            logger.error(f"Error enriching content: {str(e)}")
            state.results[self.id] = {"error": str(e)}
            state.mark_status(self.id, NodeStatus.ERROR)
            return None

class AnalysisNode(LLMNode):
    """Node that analyzes collected content."""
    
    async def process(self, state: NodeState) -> Optional[str]:
        try:
            collected_content = state.results.get("collect", {}).get("content", [])
            logger.info(f"Analyzing {len(collected_content)} content items")
            
            # Debug log the raw content
            logger.info("Raw collected content:")
            for item in collected_content:
                logger.info(f"Content item: {json.dumps(item, indent=2)}")
            
            # Group content by type/source for better analysis
            grouped_content = self._group_content(collected_content)
            
            # Debug log the grouped content
            logger.info("Grouped content:")
            for category, items in grouped_content.items():
                logger.info(f"{category}: {len(items)} items")
                for item in items:
                    logger.info(f"  - {item.get('content', '')[:100]}...")
            
            # Format content for analysis
            formatted_content = []
            for category, items in grouped_content.items():
                if items:  # Only add categories with items
                    formatted_content.append(f"\n## {category.upper()}")
                    for item in items:
                        content = item.get("content", "").strip()
                        if content:
                            formatted_content.append(f"\n{content}")
                        for url in item.get("urls", []):
                            formatted_content.append(f"[Source]({url})")
            
            # Set the prompt - LLMNode will handle the actual LLM call
            formatted_prompt = ANALYSIS_PROMPT.format(
                collect={
                    "content": "\n".join(formatted_content),
                    "date_range": state.results["collect"]["date_range"]
                }
            )
            
            # Debug log the prompt
            logger.info(f"Analysis prompt:\n{formatted_prompt}")
            
            self.prompt = formatted_prompt
            
            # Use parent class process method which handles LLM interaction
            try:
                result = await super().process(state)
                logger.info(f"Analysis result: {result}")
            except Exception as e:
                logger.error(f"Error in LLM processing: {str(e)}", exc_info=True)
                raise
            
            if result:
                # Store analysis results
                state.results[self.id] = {
                    "content_groups": grouped_content,
                    "content_analyzed": len(collected_content),
                    "response": result
                }
                state.mark_status(self.id, NodeStatus.COMPLETED)
                return "select"  # Move to content selection if successful
            
            state.mark_status(self.id, NodeStatus.ERROR)
            return None  # Stop if there's an error
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}", exc_info=True)
            state.results[self.id] = {
                "error": str(e),
                "error_details": {
                    "traceback": traceback.format_exc(),
                    "state": state.results
                }
            }
            state.mark_status(self.id, NodeStatus.ERROR)
            return None
    
    def _group_content(self, content: List[Dict]) -> Dict:
        """Group content by type for better analysis."""
        grouped = {
            "discussions": [],
            "shared_links": [],
            "announcements": [],
            "technical_content": []
        }
        
        for item in content:
            try:
                # Extract content from either direct content or embeds
                content_text = item.get("content", "")
                embeds = item.get("embeds", [])
                
                # Process embeds
                for embed in embeds:
                    if isinstance(embed, dict):
                        if embed.get("description"):
                            content_text += "\n" + embed["description"]
                        if embed.get("title"):
                            content_text += "\n" + embed["title"]
                
                # Clean and normalize content
                clean_content = content_text.lower()
                
                # Extract URLs from content and embeds
                urls = []
                if "http" in content_text:
                    # Simple URL extraction
                    for word in content_text.split():
                        if word.startswith(("http://", "https://")):
                            urls.append(word)
                
                # Add URLs from embeds
                for embed in embeds:
                    if isinstance(embed, dict) and embed.get("url"):
                        urls.append(embed["url"])
                
                # Create normalized item
                normalized_item = {
                    "content": content_text,
                    "clean_content": clean_content,
                    "urls": urls,
                    "timestamp": item.get("timestamp"),
                    "author": item.get("author"),
                    "embeds": embeds
                }
                
                # Categorize based on content characteristics
                if urls:
                    grouped["shared_links"].append(normalized_item)
                if "announcement" in clean_content:
                    grouped["announcements"].append(normalized_item)
                if any(tech in clean_content 
                      for tech in ["code", "api", "model", "implementation", "ai", "ml", "algorithm"]):
                    grouped["technical_content"].append(normalized_item)
                # Add to discussions if it doesn't fit other categories
                if not any(normalized_item in group for group in grouped.values()):
                    grouped["discussions"].append(normalized_item)
                
            except Exception as e:
                logger.error(f"Error processing content item: {str(e)}")
                logger.error(f"Problematic item: {item}")
                continue
        
        return grouped

class DraftNode(LLMNode):
    """Node that drafts newsletter content."""
    
    async def process(self, state: NodeState) -> Optional[str]:
        try:
            analysis = state.results.get("analyze", {})
            if not analysis:
                logger.warning("No analysis available for draft")
                state.mark_status(self.id, NodeStatus.ERROR)
                return None
            
            # Set prompt for LLMNode processing
            self.prompt = DRAFT_PROMPT.format(
                analyze={
                    "response": analysis["response"],
                    "content_groups": analysis.get("content_groups", {}),
                    "total_analyzed": analysis.get("content_analyzed", 0)
                }
            )
            
            # Use parent class process method which handles LLM interaction
            result = await super().process(state)
            
            if result:
                state.mark_status(self.id, NodeStatus.COMPLETED)
                return "enrich"  # Move to link enrichment if successful
            
            state.mark_status(self.id, NodeStatus.ERROR)
            return None  # Stop if there's an error
            
        except Exception as e:
            logger.error(f"Error in draft: {str(e)}")
            state.results[self.id] = {"error": str(e)}
            state.mark_status(self.id, NodeStatus.ERROR)
            return None

class ContentSelectionNode(BinaryDecisionNode):
    """Node that evaluates and selects high-quality content for the newsletter."""
    
    def __init__(self, **data):
        super().__init__(**data)
        self.prompt = """Evaluate this content for inclusion in the newsletter:

Content to evaluate:
{content}

Consider:
1. Relevance to target audience ({target_audience})
2. Technical depth and accuracy
3. Practical applicability
4. Novelty and uniqueness
5. Potential impact

Should this content be included in the newsletter?
Respond with only 'yes' or 'no'."""

    async def process(self, state: NodeState) -> Optional[str]:
        """Process and select content."""
        try:
            # Get analyzed content
            analysis = state.results.get("analyze", {}).get("response")
            if not analysis:
                logger.warning("No analysis available for content selection")
                state.mark_status(self.id, NodeStatus.ERROR)
                return None
            
            # Format prompt with content and audience
            self.prompt = self.prompt.format(
                content=analysis,
                target_audience=TARGET_AUDIENCE
            )
            
            # Use parent class process method for decision
            result = await super().process(state)
            
            if result:
                state.mark_status(self.id, NodeStatus.COMPLETED)
                return result  # Will be "yes" or "no" based on decision
            
            state.mark_status(self.id, NodeStatus.ERROR)
            return None
            
        except Exception as e:
            logger.error(f"Error in content selection: {str(e)}")
            state.results[self.id] = {"error": str(e)}
            state.mark_status(self.id, NodeStatus.ERROR)
            return None

class EditorialReviewNode(LLMNode):
    """Node that reviews and refines newsletter content."""
    
    def __init__(self, **data):
        super().__init__(**data)
        self.prompt = """Review and refine this newsletter draft for publication:

Draft Content:
{content}

Example Newsletter Format:
{example}

Guidelines:
1. Maintain consistent style and tone
2. Ensure proper markdown formatting
3. Verify all links are properly formatted
4. Check section organization matches example
5. Polish language while preserving technical accuracy
6. Keep the content engaging and accessible

Return the refined content in proper markdown format."""

    async def process(self, state: NodeState) -> Optional[str]:
        """Process and refine the newsletter content."""
        try:
            # Get draft content
            draft = state.results.get("enrich", {}).get("response")
            if not draft:
                logger.warning("No draft content to review")
                return None
            
            # Format prompt with content and example
            state.context.metadata.update({
                "content": draft,
                "example": EXAMPLE_NEWSLETTER
            })
            
            # Use parent class process method for LLM interaction
            result = await super().process(state)
            
            if result:
                # Store refined content
                state.results[self.id] = {
                    "response": state.results[self.id]["response"],
                    "original": draft
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in editorial review: {str(e)}")
            state.results[self.id] = {"error": str(e)}
            return None

def create_newsletter_prompt(additional_context: str = "") -> str:
    """
    Create a complete newsletter prompt with optional additional context.
    
    Args:
        additional_context (str): Additional instructions or context to include
    
    Returns:
        str: Complete formatted prompt
    """
    return f"""{NEWSLETTER_IDENTITY}
{NEWSLETTER_STRUCTURE}
{WRITING_STYLE}
{additional_context if additional_context else ""}"""

FORMAT_PROMPT = """Format this content into an engaging Improbable Automata newsletter:

Content: {draft}

{newsletter_template}

Important:
- Ensure ALL sections maintain relevant links
- Weekly Signal should reference specific examples from Field Notes
- Keep links natural within the flow of text"""

def create_newsletter_workflow() -> Graph:
    """Create the newsletter generation workflow."""
    workflow = Graph()
    
    # Create runtime config
    runtime_config = RuntimeConfig(
        provider="openpipe",
        persona=PersonaConfig(**PERSONA)
    )
    
    # Create nodes with runtime config
    collector = ContentCollectorNode(
        id="collect",
        channels=CHANNELS,
        categories=CATEGORIES
    )
    
    analyzer = AnalysisNode(
        id="analyze",
        runtime_config=runtime_config
    )
    
    selector = ContentSelectionNode(
        id="select",
        runtime_config=runtime_config,
        next_nodes={
            "yes": "draft",
            "no": "collect",  # Loop back for more content if rejected
            "error": None
        }
    )
    
    drafter = DraftNode(
        id="draft",
        runtime_config=runtime_config
    )
    
    enricher = LinkEnricherNode(
        id="enrich",
        runtime_config=runtime_config
    )
    
    reviewer = EditorialReviewNode(
        id="review",
        runtime_config=runtime_config
    )
    
    formatter = NewsletterFormatterNode(
        id="format",
        runtime_config=runtime_config
    )
    
    # Add nodes to graph
    workflow.add_node(collector)
    workflow.add_node(analyzer)
    workflow.add_node(selector)
    workflow.add_node(drafter)
    workflow.add_node(enricher)
    workflow.add_node(reviewer)
    workflow.add_node(formatter)
    
    # Add edges
    workflow.add_edge("collect", "default", "analyze")
    workflow.add_edge("analyze", "default", "select")
    workflow.add_edge("select", "yes", "draft")
    workflow.add_edge("select", "no", "collect")
    workflow.add_edge("draft", "default", "enrich")
    workflow.add_edge("enrich", "default", "review")
    workflow.add_edge("review", "default", "format")
    workflow.add_edge("format", "default", None)  # Terminal node
    
    # Add entry point
    workflow.add_entry_point("start", "collect")
    
    return workflow

async def main(additional_context: str = ""):
    """Initialize and run the newsletter workflow."""
    try:
        # Load environment
        from dotenv import load_dotenv
        load_dotenv()
        
        # Validate channel configuration
        if not all(CHANNELS.values()):
            missing = [name for name, id in CHANNELS.items() if not id]
            raise ValueError(f"Missing channel IDs for: {', '.join(missing)}")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        
        # Enable debug logging for workflow
        logging.getLogger("babelgraph.core.graph.base").setLevel(logging.DEBUG)
        logging.getLogger("babelgraph.core.tools.discord_tool").setLevel(logging.WARNING)
        
        logger.info("🚀 Starting newsletter workflow...")
        
        # Create workflow with debug enabled
        workflow = create_newsletter_workflow()
        workflow.debug = True  # Enable detailed state logging
        
        # Initialize state
        start_time = datetime.now()
        state = NewNodeState(
            context=NodeContext(
                metadata={"start_time": start_time.isoformat()}
            )
        )
        
        # Run workflow
        logger.info("⚙️ Running workflow...")
        final_state = await workflow.run("start", state=state)
        
        # Check workflow completion
        if not final_state:
            logger.error("❌ Workflow failed - no final state returned")
            return
            
        logger.info("✅ Workflow completed")
        
        # Print results
        print("\n📊 PIPELINE RESULTS")
        print("="*80)
        
        if "collect" in final_state.results:
            print("\n🔍 COLLECTION PHASE")
            print("-"*40)
            collect_results = final_state.results["collect"]
            print(f"Total messages processed: {collect_results['total_messages']}")
            print("\nChannel Statistics:")
            for channel, count in collect_results["channel_stats"].items():
                print(f"  #{channel}: {count} messages")
            print(f"\nDate range: {collect_results['date_range']}")
            print(f"Total links collected: {len(collect_results['link_collection'])}")
        
        # Print status of each phase with more detail
        print("\n📋 PHASE STATUS")
        print("-"*40)
        for node_id, status in final_state.status.items():
            print(f"{node_id}: {status}")
            if status == NodeStatus.ERROR and node_id in final_state.results:
                print(f"  ❌ Error: {final_state.results[node_id].get('error', 'Unknown error')}")
                # Print full error details if available
                if "error_details" in final_state.results[node_id]:
                    print(f"  Details: {final_state.results[node_id]['error_details']}")
            elif status == NodeStatus.COMPLETED and node_id in final_state.results:
                if "response" in final_state.results[node_id]:
                    print(f"  ✅ Response available")
                    # Print first 200 chars of response for debugging
                    response = final_state.results[node_id]["response"]
                    if isinstance(response, str):
                        print(f"  Preview: {response[:200]}...")
                else:
                    print(f"  ⚠️ No response data")
        
        if "format" in final_state.results:
            print("\n🔄 FINAL NEWSLETTER")
            print("="*80)
            content = final_state.results["format"]["response"]
            print(content)
            
            # Save to file
            filepath = save_newsletter_to_file(content, start_time)
            print(f"\n💾 Newsletter saved to: {filepath}")
        else:
            logger.error("❌ No formatted content available - workflow may have failed")
            # Print last available state for debugging
            print("\n🔍 LAST AVAILABLE STATE:")
            for node_id, result in final_state.results.items():
                print(f"\n{node_id}:")
                if "error" in result:
                    print(f"  ❌ Error: {result['error']}")
                    if "error_details" in result:
                        print(f"  Details: {result['error_details']}")
                elif "response" in result:
                    print(f"  ✅ Response: {result['response'][:200]}...")
                else:
                    print(f"  ℹ️ Raw result: {result}")
        
    except Exception as e:
        logger.error(f"Error running newsletter workflow: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())