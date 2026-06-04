"""
Cloudflare Workers entry point for Atlético Intelligence FastAPI.
"""

from workers import WorkerEntrypoint
import asgi

from app_worker import app


class Default(WorkerEntrypoint):
    """Cloudflare Workers ASGI handler."""

    async def fetch(self, request):
        return await asgi.fetch(app, request, self.env)
