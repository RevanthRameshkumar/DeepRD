import os
import json
import pickle
import hashlib
import threading
from typing import Any, Dict, List

class LLMCache:
    _save_lock = threading.Lock()
    def __init__(self,
                 cache_file: str = "llm_cache.pkl",
                 ignore_keys: List[str] = None,
                 backup_cache: str = None):
        """
        Initialize the cache. If a cache file exists, load it; otherwise, start with an empty cache.
        """
        self.cache_file = cache_file
        self.backup_cache_file = backup_cache
        self.ignore_keys = ignore_keys

        # Load primary cache
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                self.cache = pickle.load(f)
        else:
            self.cache = {}

        # load backup cache, if provided
        if self.backup_cache_file:
            if os.path.exists(self.backup_cache_file):
                with open(self.backup_cache_file, "rb") as f:
                    self.backup_cache = pickle.load(f)
            else:
                self.backup_cache = {}
        else:
            self.backup_cache = None

    def _generate_key(self, model_params: Dict, messages: List[Dict]) -> str:
        if self.ignore_keys:
            params = {k: v for k, v in model_params.items() if k not in self.ignore_keys}
        else:
            params = model_params
        key_data = {"model_params": params, "messages": messages}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    def get(self, model_params: Dict, messages: List[Dict]) -> Any:
        key = self._generate_key(model_params, messages)
        result = self.cache.get(key)
        if result is None and self.backup_cache is not None:
            result = self.backup_cache.get(key)
        return result

    def set(self, model_params: Dict, messages: List[Dict], result: Any) -> None:
        key = self._generate_key(model_params, messages)
        self.cache[key] = result
        self._save_cache()

    def _save_cache(self):
        """
        Persist the current cache to disk.
        If another thread is writing right now, this call will wait until that
        write completes, then grab the lock and write itself.
        """
        # This will block until the lock is free
        with LLMCache._save_lock:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)