import os
import uuid
from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase


class RoutingMemory:
    """Manages graph-based routing memory and historical episodes in Neo4j."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "localdevpassword")

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
        )
        self.driver.verify_connectivity()

    def close(self):
        """Close driver connection."""
        if self.driver:
            self.driver.close()

    def seed(self):
        """Seed known intents, specialists, and historical routing episodes."""
        specialists = [
            {"name": "whatsapp", "desc": "WhatsApp messaging, notifications, direct chats, business communication, delivery alerts"},
            {"name": "instagram", "desc": "Instagram posts, stories, reels, comments, DMs, social media engagement and marketing"},
            {"name": "websearch", "desc": "Live web search, internet research, online documentation lookup, public web articles"}
        ]

        for s in specialists:
            self.driver.execute_query(
                """
                MERGE (spec:Specialist {name: $name})
                SET spec.description = $desc,
                    spec.updated_at = datetime()
                """,
                **s,
                database_="neo4j",
            )

        intents = [
            {
                "intent": "whatsapp_messaging",
                "specialist": "whatsapp",
                "examples": [
                    "Send a WhatsApp message with the order confirmation to the customer.",
                    "Can you send a WhatsApp alert about scheduled delivery?",
                    "Send an urgent WhatsApp notification with the verification code.",
                    "Broadcast a WhatsApp update to our subscriber chat group."
                ],
            },
            {
                "intent": "instagram_engagement",
                "specialist": "instagram",
                "examples": [
                    "Publish our latest product showcase photo to Instagram with hashtags.",
                    "Check the comments and DMs on our latest Instagram reel.",
                    "Schedule an Instagram story announcement for the product launch.",
                    "Analyze engagement metrics on our recent Instagram campaign."
                ],
            },
            {
                "intent": "web_research",
                "specialist": "websearch",
                "examples": [
                    "Search the web for the latest Python performance benchmarks.",
                    "Find the official documentation and best practices for modern web frameworks.",
                    "Look up current weather forecast and news updates online.",
                    "Search the internet for reviews and comparison between leading cloud providers."
                ],
            },
        ]

        for item in intents:
            intent_name = item["intent"]
            specialist_name = item["specialist"]
            self.driver.execute_query(
                """
                MERGE (i:Intent {name: $intent})
                MERGE (s:Specialist {name: $specialist})
                MERGE (i)-[r:ROUTES_TO]->(s)
                SET r.source = 'seed',
                    r.updated_at = datetime()
                """,
                intent=intent_name,
                specialist=specialist_name,
                database_="neo4j",
            )

            for ex in item["examples"]:
                self.driver.execute_query(
                    """
                    MERGE (e:RoutingEpisode {example: $example})
                    SET e.intent = $intent,
                        e.selected = $specialist,
                        e.urgency = 'normal',
                        e.outcome = 'seeded',
                        e.success = true,
                        e.created_at = datetime()
                    WITH e
                    MATCH (i:Intent {name: $intent})
                    MATCH (s:Specialist {name: $specialist})
                    MERGE (e)-[:SUPPORTS]->(i)
                    MERGE (e)-[:SELECTED]->(s)
                    """,
                    example=ex,
                    intent=intent_name,
                    specialist=specialist_name,
                    database_="neo4j",
                )

    def retrieve(self, intent: str) -> List[Dict[str, Any]]:
        """Fetch known routes and supporting examples for a given intent."""
        records, _, _ = self.driver.execute_query(
            """
            MATCH (i:Intent {name: $intent})-[r:ROUTES_TO]->(s:Specialist)
            OPTIONAL MATCH (e:RoutingEpisode)-[:SUPPORTS]->(i)
            RETURN i.name AS intent,
                   s.name AS specialist,
                   r.source AS source,
                   collect(DISTINCT coalesce(e.query, e.example))[0..5] AS examples
            """,
            intent=intent,
            database_="neo4j",
        )
        return [record.data() for record in records]

    def find_relevant_memory(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Search Neo4j memory for relevant routes and past routing episodes matching query words."""
        # Clean query tokens
        stop_words = {"the", "a", "an", "is", "are", "and", "or", "to", "for", "in", "of", "on", "with", "my", "our", "i", "they"}
        words = [w.strip(".,?!\"'").lower() for w in query.split() if len(w) > 2 and w.lower() not in stop_words]

        # 1. Fetch all known intent -> specialist routes
        routes_records, _, _ = self.driver.execute_query(
            """
            MATCH (i:Intent)-[r:ROUTES_TO]->(s:Specialist)
            RETURN i.name AS intent, s.name AS specialist, s.description AS description
            """,
            database_="neo4j",
        )
        all_routes = [r.data() for r in routes_records]

        # 2. Find matching past episodes based on query keywords or text containment
        episodes_records, _, _ = self.driver.execute_query(
            """
            MATCH (e:RoutingEpisode)-[:SELECTED]->(s:Specialist)
            OPTIONAL MATCH (e)-[:SUPPORTS]->(i:Intent)
            RETURN coalesce(e.query, e.example) AS text,
                   e.selected AS selected,
                   coalesce(e.intent, i.name, 'unknown') AS intent,
                   e.urgency AS urgency,
                   e.confidence AS confidence,
                   e.outcome AS outcome,
                   e.success AS success,
                   e.created_at AS created_at
            ORDER BY e.created_at DESC
            LIMIT 50
            """,
            database_="neo4j",
        )

        matched_episodes = []
        for ep in episodes_records:
            ep_data = ep.data()
            text = (ep_data.get("text") or "").lower()
            # Simple match score based on shared words
            score = sum(1 for w in words if w in text)
            if score > 0:
                ep_data["relevance_score"] = score
                matched_episodes.append(ep_data)

        # Sort by relevance score descending
        matched_episodes.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        top_episodes = matched_episodes[:limit]

        # Match intents
        matched_intents = []
        for route in all_routes:
            intent_name = route["intent"].lower().replace("_", " ")
            specialist_name = route["specialist"].lower()
            desc = (route.get("description") or "").lower()
            if any(w in intent_name or w in specialist_name or w in desc for w in words):
                matched_intents.append(route)

        summary_parts = []
        if matched_intents:
            intents_str = ", ".join([f"{r['intent']} -> {r['specialist']}" for r in matched_intents])
            summary_parts.append(f"Known routes matched: [{intents_str}]")
        if top_episodes:
            ex_summary = "; ".join([f"'{ep['text'][:50]}...' -> {ep['selected']}" for ep in top_episodes[:2]])
            summary_parts.append(f"Historical episodes: [{ex_summary}]")

        summary = " | ".join(summary_parts) if summary_parts else "No direct historical match; defaulting to semantic routing."

        return {
            "all_routes": all_routes,
            "matched_intents": matched_intents,
            "matched_episodes": top_episodes,
            "summary": summary,
        }

    def record_outcome(
        self,
        query: str,
        intent: str,
        selected: str,
        urgency: str = "normal",
        confidence: float = 0.0,
        execution_time_ms: float = 0.0,
        success: bool = True,
        reasoning: str = "",
    ) -> str:
        """Record an executed routing decision as a new RoutingEpisode in Neo4j."""
        episode_id = str(uuid.uuid4())
        self.driver.execute_query(
            """
            MERGE (i:Intent {name: $intent})
            MERGE (s:Specialist {name: $selected})
            CREATE (e:RoutingEpisode {
                id: $id,
                query: $query,
                intent: $intent,
                selected: $selected,
                urgency: $urgency,
                confidence: $confidence,
                execution_time_ms: $execution_time_ms,
                success: $success,
                reasoning: $reasoning,
                outcome: 'executed',
                created_at: datetime()
            })
            MERGE (e)-[:SUPPORTS]->(i)
            MERGE (e)-[:SELECTED]->(s)
            """,
            id=episode_id,
            query=query,
            intent=intent,
            selected=selected,
            urgency=urgency,
            confidence=confidence,
            execution_time_ms=execution_time_ms,
            success=success,
            reasoning=reasoning,
            database_="neo4j",
        )
        return episode_id

    def update_feedback(self, episode_id: str, success: bool, feedback: str = "") -> bool:
        """Update an episode with user feedback."""
        records, _, _ = self.driver.execute_query(
            """
            MATCH (e:RoutingEpisode {id: $id})
            SET e.success = $success,
                e.feedback = $feedback,
                e.updated_at = datetime()
            RETURN e.id AS id
            """,
            id=episode_id,
            success=success,
            feedback=feedback,
            database_="neo4j",
        )
        return len(records) > 0

    def get_recent_episodes(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent routing episodes for inspection and audit."""
        records, _, _ = self.driver.execute_query(
            """
            MATCH (e:RoutingEpisode)
            OPTIONAL MATCH (e)-[:SELECTED]->(s:Specialist)
            OPTIONAL MATCH (e)-[:SUPPORTS]->(i:Intent)
            RETURN e.id AS id,
                   coalesce(e.query, e.example) AS query,
                   coalesce(e.intent, i.name) AS intent,
                   coalesce(e.selected, s.name) AS selected,
                   e.urgency AS urgency,
                   e.confidence AS confidence,
                   e.execution_time_ms AS execution_time_ms,
                   e.success AS success,
                   e.outcome AS outcome,
                   e.reasoning AS reasoning,
                   toString(e.created_at) AS created_at
            ORDER BY e.created_at DESC
            LIMIT $limit
            """,
            limit=limit,
            database_="neo4j",
        )
        return [record.data() for record in records]

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics of Neo4j routing memory."""
        records, _, _ = self.driver.execute_query(
            """
            OPTIONAL MATCH (i:Intent) WITH count(DISTINCT i) AS intents_count
            OPTIONAL MATCH (s:Specialist) WITH intents_count, count(DISTINCT s) AS specialists_count
            OPTIONAL MATCH (e:RoutingEpisode) WITH intents_count, specialists_count, count(DISTINCT e) AS episodes_count
            OPTIONAL MATCH (e_succ:RoutingEpisode {success: true})
            RETURN intents_count, specialists_count, episodes_count, count(DISTINCT e_succ) AS successful_episodes
            """,
            database_="neo4j",
        )
        if records:
            d = records[0].data()
            total = d.get("episodes_count", 0)
            succ = d.get("successful_episodes", 0)
            success_rate = (succ / total * 100) if total > 0 else 100.0
            return {
                "intents_count": d.get("intents_count", 0),
                "specialists_count": d.get("specialists_count", 0),
                "episodes_count": total,
                "successful_episodes": succ,
                "success_rate": round(success_rate, 1),
            }
        return {"intents_count": 0, "specialists_count": 0, "episodes_count": 0, "successful_episodes": 0, "success_rate": 0.0}

    def clear_memory(self):
        """Clear all nodes in the database (use with caution)."""
        self.driver.execute_query(
            "MATCH (n) DETACH DELETE n",
            database_="neo4j",
        )