from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
WEB_DIRECTORY = "./web"

# New endpoint and new class IDs: safe to install alongside the old extension.
try:
    from aiohttp import web
    from server import PromptServer
    from .nodes import _options, _DIR

    @PromptServer.instance.routes.get("/h3img8_v2/identities")
    async def h3img8_v2_identities(request):
        files = [x for x in _options() if x != "NO_IDENTITIES_YET.safetensors"]
        return web.json_response({"files": files, "directory": str(_DIR)})
except Exception as e:
    print(f"[H3Img8V2] Refresh endpoint error: {e!r}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
