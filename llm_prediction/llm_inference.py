import os
import time
import functools
import asyncio
import re
import unicodedata
from enum import Enum
from copy import deepcopy
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from dotenv import load_dotenv
from llm_cache import LLMCache

# Load environment variables from .env file
load_dotenv()

OPENAI_KEY = os.getenv('OPENAI_API_KEY')
TOGETHER_KEY = os.getenv('TOGETHER_API_KEY')

class Provider(Enum):
    DEEPINFRA = 1
    OPENAI = 2
    TOGETHER = 3


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to execute.")
        return result
    return wrapper

class CachingLLM:
    _shared_loop = None  # Class-level event loop to prevent leaks

    def __init__(self,
                 model_params,
                 initial_messages,
                 cache_file,
                 cache_only=False,
                 skip_cache=False,
                 rewrite_cache=False,
                 cache_skip_fn = None):

        self.model_params = deepcopy(model_params)
        self.provider = None
        if model_params['model'] == 'deepseek-ai/DeepSeek-R1':
            self.provider = Provider.TOGETHER
        elif model_params['model'] == 'deepseek-ai/DeepSeek-V3':
            self.provider = Provider.TOGETHER
        elif model_params['model'] == 'gpt-4o-mini':
            self.provider = Provider.OPENAI
        elif model_params['model'] == 'gpt-4o':
            self.provider = Provider.OPENAI
        elif model_params['model'] == 'o3-mini':
            self.provider = Provider.OPENAI
        elif model_params['model'] == 'o3':
            self.provider = Provider.OPENAI
        elif model_params['model'] == 'gpt-4':
            self.provider = Provider.OPENAI
        elif model_params['model'] == 'gpt-4o':
            self.provider = Provider.OPENAI
        else:
            raise NotImplementedError('model {} not implemented'.format(model_params['model']))

        # Lazily initialize network client to avoid creating it when using cache-only
        self.client = None

        self.cache_skip_fn = cache_skip_fn
        self.skip_cache = skip_cache
        self.cache_only = cache_only
        self.rewrite_cache = rewrite_cache
        self.cache_file = cache_file
        self.cache = LLMCache(ignore_keys=['max_tokens'], cache_file=cache_file, backup_cache='llm_cache_backup_4.pkl')

        self.initial_messages = deepcopy(initial_messages)

        self.active_queries = 0

        self.total_successful=0

    def _ensure_client(self):
        if self.client is not None:
            return
        if self.provider == Provider.DEEPINFRA:
            self.client = AsyncOpenAI(api_key=DEEPINFRA_KEY, base_url="https://api.deepinfra.com/v1/openai")
        elif self.provider == Provider.OPENAI:
            self.client = AsyncOpenAI(api_key=OPENAI_KEY)
        elif self.provider == Provider.TOGETHER:
            self.client = AsyncOpenAI(api_key=TOGETHER_KEY, base_url="https://api.together.xyz/v1")
        else:
            raise NotImplementedError('provider {} not implemented'.format(self.provider))


    def _skip_cache_from_function(self, cached_result):
        if isinstance(cached_result, ChatCompletion):
            content = cached_result.choices[0].message.content.strip()
        else:
            content = cached_result
        if self.cache_skip_fn is not None and content is not None:
            if self.cache_skip_fn(content):
                print('skipped via function')
                return True

        return False

    async def generate_response(self, input_message, **kwargs):
        start_time = time.perf_counter()

        if input_message is None:
            return None

        previous_messages = kwargs.get('previous_messages', [])
        new_message_list = self.initial_messages + previous_messages + [{"role": "user", "content": input_message}]
        cached_result = self.cache.get(self.model_params, new_message_list)
        if cached_result is not None and not(self.skip_cache) and not(self._skip_cache_from_function(cached_result)) and self.rewrite_cache is False:
            _dbg("cache hit")

            elapsed_time = time.perf_counter() - start_time
            _dbg(f"LLM call took {elapsed_time:.4f} seconds")
            self.total_successful += 1
            _dbg(f"Total Successful: {self.total_successful}")
            return cached_result
        elif self.cache_only:
            return None
        else:
            if (self.model_params['model'] == 'deepseek-ai/DeepSeek-R1' and self.provider == Provider.TOGETHER) or \
                self.model_params['model'] == 'gpt-4' or \
                self.model_params['model'] == 'o3' or True:
                delay = 20

                while True:
                    if self.active_queries > 100:
                        await asyncio.sleep(1)
                    else:
                        _dbg('starting query')
                        break
            else:
                delay = 5
            max_retries = 5

            if new_message_list[-1] == '<END>':
                return

            for attempt in range(max_retries):
                start_time_attempt = time.perf_counter()
                try:

                    self.active_queries += 1
                    _dbg('sent query')
                    self._ensure_client()
                    response = await self.client.chat.completions.create(
                        **self.model_params,
                        messages=new_message_list
                    )
                    _dbg('got response')
                    self.active_queries -= 1

                    llm_response = response

                    if not(self.skip_cache):
                        self.cache.set(self.model_params, new_message_list, llm_response)
                        _dbg("Cache miss, result stored:")
                    _dbg(llm_response)

                    elapsed_time = time.perf_counter() - start_time
                    elapsed_time_attempt = time.perf_counter() - start_time_attempt
                    _dbg(f"LLM call took {elapsed_time:.4f} seconds total")
                    _dbg(f"LLM call took {elapsed_time_attempt:.4f} seconds for attempt")
                    self.total_successful += 1
                    _dbg(f"Total Successful: {self.total_successful}")
                    return llm_response
                except Exception as e:
                    _dbg(f"Error on attempt {attempt + 1}/{max_retries}: {e}")
                    await asyncio.sleep(delay)

    async def _async_generate_responses(self, input_messages, previous_messages):
        """
        Asynchronously processes a list of input messages concurrently.
        """
        if self.provider == Provider.TOGETHER:
            tasks = [asyncio.create_task(self.generate_response(msg, offset=i, previous_messages=previous_messages_list)) for i, (msg, previous_messages_list) in enumerate(zip(input_messages, previous_messages))]
        else:
            tasks = [asyncio.create_task(self.generate_response(msg, previous_messages=previous_messages_list)) for msg, previous_messages_list in zip(input_messages, previous_messages)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        return responses

    def generate_responses(self, input_messages, previous_messages=None):
        """
        Synchronous wrapper that processes a list of input messages concurrently
        using asyncio under the hood.
        """
        if previous_messages is None:
            previous_messages = [[] for _ in input_messages]

        # Reuse class-level event loop to prevent task accumulation and leaks
        if CachingLLM._shared_loop is None or CachingLLM._shared_loop.is_closed():
            CachingLLM._shared_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(CachingLLM._shared_loop)

        loop = CachingLLM._shared_loop
        return loop.run_until_complete(self._async_generate_responses(input_messages, previous_messages))

def get_content_from_response(llm_response):
    if llm_response is None:
        return None
    if isinstance(llm_response, str):
        content = llm_response
    else:
        content = llm_response['choices'][0]['message']['content'].strip()

    # Strip DeepSeek/other "thinking" blocks like <think> ... </think>
    if content:
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL|re.IGNORECASE)
        # Remove any stray tags if present
        content = re.sub(r"</?think>", "", content, flags=re.IGNORECASE)
        content = content.strip()
    return content

def get_path_extraction_content_from_response(llm_response):
    if llm_response is None:
        return ''
    if isinstance(llm_response, str):
        raise ValueError
    model = llm_response['model']

    _dbg(model)
    # Always strip <think>...</think> blocks if present
    content = llm_response['choices'][0]['message']['content'].strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL|re.IGNORECASE)
    content = re.sub(r"</?think>", "", content, flags=re.IGNORECASE)
    return content.strip()

def get_start_and_end_node_from_question(question):
    # Regex pattern to capture the start and end node numbers
    pattern = r"node\s+(\d+)\s+and\s+node\s+(\d+)"

    # Search for the pattern in the text
    match = re.search(pattern, question)
    if match:
        start_node = int(match.group(1))
        end_node = int(match.group(2))
    else:
        raise ValueError("No match found")

    return start_node, end_node

def make_SI_question(start_node, end_node):
    question = \
"""
Given the following start and end node, determine if there is a path connecting them:

Start node: {}
Goal node: {}
""".format(start_node, end_node)

    return question

def make_adjacency_dict(edge_list):
    adjacency_dict = {}
    for u, v in edge_list:
        # Add the edge u -> v
        if u not in adjacency_dict:
            adjacency_dict[u] = []
        adjacency_dict[u].append(v)

        # For an undirected graph, also add the edge v -> u
        if v not in adjacency_dict:
            adjacency_dict[v] = []
        adjacency_dict[v].append(u)
    return adjacency_dict

def check_end_condition(previous_message_list):
    if previous_message_list[-1] == '<END>':
        return True
    return False

def add_messages_to_previous_messages(previous_messages, messages, usages):
    for message, previous_message_list, usage_list in zip(messages, previous_messages, usages):
        if message is None:
            continue
        previous_message_list.append({"role": "user", "content": message})
        usage_list.append(None)

def check_get_edges(text):
    return re.search(r'get_edges\(\s*(\d+)\s*\)', text)


def recursive_to_dict(obj):
    # If it's a dictionary, convert each key/value recursively
    if isinstance(obj, dict):
        return {key: recursive_to_dict(value) for key, value in obj.items()}

    # If it's a list, tuple, or set, convert each item recursively
    elif isinstance(obj, (list, tuple, set)):
        return [recursive_to_dict(item) for item in obj]

    # If it has a __dict__ attribute, convert that dictionary recursively
    elif hasattr(obj, '__dict__'):
        return recursive_to_dict(vars(obj))

    # Otherwise, return the object as-is (e.g., int, str, etc.)
    else:
        return obj

def update_previous_message_list(i, previous_message_list, llm_response, adjacency_dicts, usage_list):
    if get_content_from_response(llm_response) is None:
        return None

    usage_list.append(recursive_to_dict(llm_response.usage))
    llm_response = get_content_from_response(llm_response)
    response_to_analyze = llm_response
    if '</think>' in llm_response:
        response_to_analyze = llm_response.split('</think>')[1]

    if '|YES|'.lower() in response_to_analyze.lower() or '|NO|'.lower() in response_to_analyze.lower():
        previous_message_list += [{"role": "assistant", "content": llm_response}]
        previous_message_list.append('<END>')
        return None
    else:
        match = check_get_edges(response_to_analyze)
        if match:
            param = int(match.group(1))
            if param in adjacency_dicts[i]:
                neighbors = adjacency_dicts[i][param] + [param]
                previous_message_list += [{"role": "assistant", "content": llm_response}]
                return {"role": "user", "content": "{} is connected to {}".format(param, ", ".join([str(n) for n in neighbors]))}
            else:
                previous_message_list += [{"role": "assistant", "content": llm_response}]
                return {"role": "user",
                        "content": "{} does not have any connections".format(param)}
        else:
            previous_message_list += [{"role": "assistant", "content": llm_response}]
            previous_message_list.append('<END>')
            return None


def sanitize_filename(filename: str, replace_with: str = "_", max_length: int = None) -> str:
    """
    Remove or replace characters not safe for filenames on most operating systems.

    - Normalizes unicode to NFKD form and strips combining marks.
    - Replaces any character not in [A-Za-z0-9._-] with `replace_with`.
    - Collapses multiple consecutive `replace_with` into one.
    - Optionally truncates the result to max_length characters.

    Args:
        filename:      The original filename string.
        replace_with:  Character to substitute for invalid chars (default "_").
        max_length:    If set, truncates the final name to this many characters.

    Returns:
        A sanitized filename string.
    """
    # 1. Normalize unicode characters (e.g. "küche" → "kuche")
    nfkd = unicodedata.normalize("NFKD", filename)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")

    # 2. Replace invalid chars with the replacement
    #    Valid chars: letters, numbers, dot, underscore, dash
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", replace_with, ascii_str)

    # 3. Collapse multiple replacements in a row
    sep = re.escape(replace_with)
    sanitized = re.sub(rf"{sep}{{2,}}", replace_with, sanitized)

    # 4. Trim leading/trailing replacement chars or dots
    sanitized = sanitized.strip(f"{replace_with}.")

    # 5. Optionally truncate to max_length
    if max_length is not None and len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip(f"{replace_with}.")

    # Fallback to a default name if it ends up empty
    return sanitized or "untitled"

LLM_VERBOSE = os.environ.get("LLM_VERBOSE", "0") == "1"

def _dbg(*args, **kwargs):
    if LLM_VERBOSE:
        print(*args, **kwargs)
