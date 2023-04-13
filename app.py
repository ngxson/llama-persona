import llama_cpp
import sys
import re
import signal
import imp
from alive_progress import alive_bar

try:
    imp.find_module('config')
    from config import *
except ImportError:
    print('Cannot find config.py')
    print('Make sure that you copy "config.example.py" to "config.py"')
    exit(1)

model = llama_cpp.Llama(
    model_path=MODEL_PATH,
    seed=SEED,
    n_threads=N_THREADS,
    last_n_tokens_size=N_LAST_TOKENS,
    n_ctx=N_CTX,
)

TOKEN_BOS = model.token_bos()
TOKEN_EOS = model.token_eos()

PROMPT_INIT = f""" {PERSONA_DESC}

Pretend that you are {PERSONA_NAME}. Below is an instruction that describes a task. Write a response that appropriately completes the request.""".encode()

is_received_stop_signal = False  # TODO: catching SIGINT signal


def init():
    global state_after_init_prompt
    print('')
    m_eval(model, m_tokenize(model, PROMPT_INIT, True), False, 'Starting up...')
    state_after_init_prompt = save_state(model)

    while True:
        print('\n> ', end='', flush=True)
        input_txt = input()
        process_user_input(input_txt)


def process_user_input(text):
    global state_after_init_prompt, is_received_stop_signal
    is_received_stop_signal = False

    # generate response
    response_bytes = b''
    response_txt = ''
    input_tokens = m_tokenize(model,
                              (f'\n\n### Instruction:\n\n{text}\n\n### Response:\n\n').encode())
    for token in m_generate(
        model,
        input_tokens,
        top_k=TOP_K,
        top_p=TOP_P,
        temp=TEMP,
        repeat_penalty=REPEAT_PENALTY,
    ):
        if token == TOKEN_EOS:
            break
        should_stop = False
        response_added_bytes = model.detokenize([token])
        response_bytes += response_added_bytes
        response_txt = response_bytes.decode()
        if '###' in response_txt:
            response_txt = re.sub(r"\s+###", '', response_txt)
            sys.stdout.write("\033[K")  # Clear to the end of line
            print(response_txt.split('\n')[-1], end='', flush=True)
            should_stop = True
        print(response_added_bytes.decode(), end='', flush=True)
        if should_stop:
            break

    # build context for next message
    input_ins_truncated = ' '.join(text.split(' ')[:N_TOKENS_KEEP_INS])
    input_res_truncated = ' '.join(
        response_txt.split(' ')[:N_TOKENS_KEEP_RES])
    input_history = f'\n\n### Instruction:\n\n{input_ins_truncated}\n\n### Response:\n\n{input_res_truncated}'

    history_tokens = m_tokenize(model, input_history.encode())
    restore_state(model, state_after_init_prompt)
    print('\n\n', end='', flush=True)
    m_eval(model, history_tokens, False, 'Build context...')

###########################################


def save_state(model: llama_cpp.Llama):
    saved_state = {}
    saved_state['tokens_consumed'] = model.tokens_consumed
    saved_state['last_n_tokens_data'] = model.last_n_tokens_data.copy()
    saved_state['kv_cache_token_count'] = llama_cpp.llama_get_kv_cache_token_count(
        model.ctx)
    saved_state['kv_cache_size'] = llama_cpp.llama_get_kv_cache_size(model.ctx)
    saved_state['kv_cache'] = (
        llama_cpp.c_uint8 * int(saved_state['kv_cache_size']))()
    curr_kv_cache = llama_cpp.llama_get_kv_cache(model.ctx)
    llama_cpp.ctypes.memmove(
        saved_state['kv_cache'], curr_kv_cache, int(saved_state['kv_cache_size']))
    return saved_state


def restore_state(model: llama_cpp.Llama, saved_state):
    model.tokens_consumed = saved_state['tokens_consumed']
    model.last_n_tokens_data = saved_state['last_n_tokens_data'].copy()
    llama_cpp.llama_set_kv_cache(
        model.ctx,
        saved_state['kv_cache'],
        saved_state['kv_cache_size'],
        saved_state['kv_cache_token_count'],
    )


###########################################


def m_generate(model: llama_cpp.Llama, tokens, top_k, top_p, temp, repeat_penalty):
    """Generate without self.reset()"""
    global is_received_stop_signal
    is_received_stop_signal = False
    while True:
        if is_received_stop_signal:
            yield TOKEN_EOS
        m_eval(model, tokens, True)
        token = model.sample(
            top_k=top_k,
            top_p=top_p,
            temp=temp,
            repeat_penalty=repeat_penalty,
        )
        tokens_or_none = yield token
        tokens = [token]
        if tokens_or_none is not None:
            tokens.extend(tokens_or_none)


def m_tokenize(model: llama_cpp.Llama, text: bytes, add_bos=False):
    assert model.ctx is not None
    n_ctx = llama_cpp.llama_n_ctx(model.ctx)
    tokens = (llama_cpp.llama_token * int(n_ctx))()
    n_tokens = llama_cpp.llama_tokenize(
        model.ctx,
        text,
        tokens,
        n_ctx,
        llama_cpp.c_bool(add_bos),
    )
    if int(n_tokens) < 0:
        raise RuntimeError(
            f'Failed to tokenize: text="{text}" n_tokens={n_tokens}')
    return list(tokens[:n_tokens])


def m_eval(model: llama_cpp.Llama, tokens, stop_on_signal=False, show_progress=False):
    global is_received_stop_signal

    def chunks(lst, n):
        return [lst[i:i + n] for i in range(0, len(lst), n)]

    batches = chunks(tokens, N_BATCH)

    def __eval(bar=None):
        global is_received_stop_signal
        for i, batch in enumerate(batches):
            if stop_on_signal and is_received_stop_signal:
                is_received_stop_signal = False
                return
            else:
                model.eval(batch)
                bar(len(batch)) if bar is not None else None

    if show_progress:
        with alive_bar(len(tokens), theme='classic', title=show_progress) as bar:
            __eval(bar)
    else:
        __eval()


init()
