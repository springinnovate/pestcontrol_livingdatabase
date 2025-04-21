import os
from diskcache import Index, Disk
import pickle, sys, zlib
import sqlite3

# sqlite3.register_adapter(memoryview, lambda mv: mv.tobytes())

# BROWSER_CACHE = Index("browser_cache")
# PROCESS_URL_CACHE = Index("process_url_cache_v3")
# SCRUBBED_OPENAI_CACHE = Index("scrubbed_by_4o_mini_v2")


# for cache_id, cache in [
#     ("process url", PROCESS_URL_CACHE),
#     ("browser_cache", BROWSER_CACHE),
#     ("scrubbed openai", SCRUBBED_OPENAI_CACHE),
# ]:
#     print(f"test {cache_id}")
#     for k, v in cache.items():
#         pass


cache_dir = "browser_cache"
idx = Index(cache_dir)

needle = "https://www.nature.com/articles/s41598-022-14625-9"
key_pickle = pickle.dumps(needle, protocol=pickle.HIGHEST_PROTOCOL)
key_hash = zlib.crc32(key_pickle)

db = sqlite3.connect(os.path.join(cache_dir, "db.sqlite"))
print(
    "bytes param :",
    db.execute(
        "SELECT 1 FROM Cache WHERE key_hash=? AND key=?", (key_hash, key_pickle)
    ).fetchone(),
)

print(
    "mv param    :",
    db.execute(
        "SELECT 1 FROM Cache WHERE key_hash=? AND key=?",
        (key_hash, memoryview(key_pickle)),
    ).fetchone(),
)
