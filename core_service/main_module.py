import asyncio
import logging
from datetime import datetime

from modules import Filter, Dispatcher, PreProcess
from client import send_request

from utils.curent_data_utils import load_data, update_data
from utils.dialodue_history import make_dialogue_history, add_dialogue_entry

logger = logging.getLogger(__name__)

class Core:
    data = {
        'modules_online': {
            'stt': False,
            'cv': False,
            'sys': False
        },
        # ПОТРЕБУЕТ ЧИСТКИ ПО ДОСТИЖЕНИЮ ОПРЕДЕЛЕННОЙ ДЛИНЫ. 
        # Надо будет написать суммаризатор для этого
        # В дальнейшей обработке я использую только последный контектс
        # Могу спокой делать краткую выжимку из половины ддмиалога и не сломаю систему даже))))))))
        'context': [ 
            # { # Just reference
            #     '/sys': '',
            #     '/cv': '',
            #     '/user': '',
            #     '/model': ''
            # }
        ],
        'input': {
            'stt': {
                'timestamp': 0,
                'last_update': 0,
                'importance': False,
                'data':{
                    "speaker": "Владелец",
                    "language": "ru",
                    'text': ""
                }
            },
            'cv': {
                'timestamp': 0,
                'last_update': 0,
                'importance': False,
                'data':{
                    'text': "",
                    'objects': []
                }
            },
            'sys': {
                'timestamp': 0,
                'last_update': 0,
                'importance': False,
                'data': {

                }
            }
        }
    }

    is_processing = False

    def __init__(self):
        logger.info("=" * 50)
        logger.info("Core module initialization started")
        try:
            self.data = load_data()
            logger.info("Current data loaded successfully.")
        except FileNotFoundError:
            logger.warning("Current data file not found. Initializing with default values.")

        try:
            logger.debug("Initializing Filter module...")
            self.filter = Filter()
            logger.debug("✓ Filter module loaded")
            
            logger.debug("Initializing Dispatcher module...")
            self.dispatcher = Dispatcher()
            logger.debug("✓ Dispatcher module loaded")
            
            logger.debug("Initializing PreProcess module...")
            self.pre_process = PreProcess()
            logger.debug("✓ PreProcess module loaded")
            
            logger.info("✓ Core modules initialized successfully")
            logger.info("Core data structure initialized | context_size=0 | modules_online={'stt': False, 'cv': False, 'sys': False}")
            logger.info("=" * 50)
        except Exception as e:
            logger.error(f"✗ Core init error: {type(e).__name__}: {e}", exc_info=True)
            raise

    def add_stt_input(self,text, speaker, language): # Updating STT input data
        logger.info(f"STT input received | speaker={speaker} | language={language} | text_len={len(text)}")
        logger.debug(f"STT input text: '{text}'")
        
        corent_timestamp = datetime.now().timestamp()
        last_timestamp = self.data['input']['stt']['timestamp']
        self.data['input']['stt']['timestamp'] = corent_timestamp
        self.data['input']['stt']['last_update'] = corent_timestamp - last_timestamp
        self.data['input']['stt']['data'] = {
            'text': text,
            'language': language,
            'speaker': speaker
        }
        logger.debug(f"STT data updated | timestamp={corent_timestamp} | context_size={len(self.data['context'])}")

        # filter
        if not self.is_processing:
            self.is_processing = True
            filter_des = self._stt_filter(self.data['input'])
            if filter_des:
                self._main_pipeline()
            else:
                logger.debug(f"STT Input rejected by filter")
            self.is_processing = False
        else:
            logger.debug(f"Previous processing still in progress")


    def _main_pipeline(self): # Checking new input importance 
        logger.debug("=== Starting input filtering pipeline ===")
        logger.debug(f"Current context size: {len(self.data['context'])} elements")
        logger.debug(f"STT text: '{self.data['input']['stt']['data']['text']}'")
        
        try:
            need_answer = self._need_answer()
            logger.info(f"Filter decision | need_answer={need_answer}")
            
            if need_answer:
                logger.info("✓ Input approved by filter - Processing...")
                logger.debug(f"Filter importance: stt={self.data['input']['stt']['importance']} | cv={self.data['input']['cv']['importance']} | sys={self.data['input']['sys']['importance']}")
                
                # Bring back when modulest will be done
                # logger.info("Requesting extra information from dispatcher")
                # ext_inf = self.dispatcher.get_ext_inf(self.data['input'])

                try:
                    final_input, final_context = self._preprocess_llm_input(self.data['input'], {})
                    logger.debug(f"Pre-processing completed | input_keys={list(final_input.keys())}")
                    
                    model_answer = self._send_request_to_llm(final_input, self.data['context'])
                    if model_answer:
                        logger.info(f"✓ Chat response received | response_len={len(str(model_answer))}")
                        logger.debug(f"Model answer: {model_answer}")
                        self._update_context(final_context, model_answer)
                        logger.info(f"Context updated | new_context_size={len(self.data['context'])}")
                    else:
                        logger.warning("⚠ Chat service returned None or empty response")
                    self._update_data(self.data)

                except Exception as e:
                    logger.error(f"✗ Chat request error: {type(e).__name__}: {e}", exc_info=True)
            else:
                logger.debug(f"✗ Input rejected by filter")
                logger.debug(f"Filter importance: stt={self.data['input']['stt']['importance']} | cv={self.data['input']['cv']['importance']} | sys={self.data['input']['sys']['importance']}")
        except Exception as e:
            logger.error(f"✗ Input check error: {type(e).__name__}: {e}", exc_info=True)

    def _update_context(self, last_context: dict, model_answer: str):
        logger.debug("Updating context with model response")
        last_context.update({'/model': model_answer})
        self.data['context'].append(last_context)
        logger.debug(f"Context updated successfully | new_total_context_size={len(self.data['context'])} | model_answer_len={len(str(model_answer))}")


    # --- Input Filter --- 
    def _need_answer(self) -> bool:
        try:
            input = self.data['input']
            context = self.data['context'][-2:]

            logger.info("🔍 Filtering input - Checking if answer is needed")
            logger.debug(f"Input STT text: '{input['stt']['data'].get('text', 'N/A')}'")
            logger.debug(f"Input CV: N/A")
            logger.debug(f"Input SYS: N/A")

            is_important = self._is_important(input, context)
            if not is_important:
                return False
            


            return True
        except Exception as e:
            logger.error(f"✗ Error in _need_answer(): {type(e).__name__}: {e}", exc_info=True)
            return False
        
    # -- General Filter -- 
    def _is_important(self, input: dict, context: list[dict]):
        try:
            importance = self.filter.get_input_importance(input, context)
            logger.debug(f"Filter importance results: {importance}")

            self._update_input_importance(input, importance)
            logger.debug(f"Input importance updated | stt={input['stt']['importance']} | cv={input['cv']['importance']} | sys={input['sys']['importance']}")

            if any(importance.values()):
                logger.info(f"✓ Answer needed | important_sources={[k for k, v in importance.items() if v]}")
                return True

            logger.debug("✗ No important input detected")
            return False
        except Exception as e:
            logger.error(f"✗ Error in _is_important(): {type(e).__name__}: {e}", exc_info=True)
            return False

    def _update_input_importance(self, input: dict, importance: dict):
        logger.debug("Updating input importance flags")
        input['stt']['importance'] = importance['stt']
        input['cv']['importance'] = importance['cv']
        input['sys']['importance'] = importance['sys']

    # -- Stt Filter --
    def _stt_filter(self, input: dict) -> bool:
        try:
            should_shut_up = self._should_shut_up(input)
            # logger.debug(f"")
            if should_shut_up:
                return False
            return True
        except Exception as e:
            logger.error(f"✗ Error in _stt_filter(): {type(e).__name__}: {e}", exc_info=True)
            return False

    def _should_shut_up(self, input: dict):
        should_shut_up = self.filter.should_shut_up(input)
        if should_shut_up: 
            # Будет отправлять запрос к TTS на то чтобы выключить 
            return True
        return False
    
    # -- Cv Filter --
    # -- Sys Filter --



    # --- Preprocessing ---
    def _preprocess_llm_input(self, input_data: dict, extra_data: list[dict]):
        logger.debug("Starting pre-processing...")
        return self.pre_process.process(input_data, extra_data)
    
    # --- LLM Request ---
    def _send_request_to_llm(self, input_data: dict, context: list[dict]):
        logger.info(f"Sending request to chat service | context_len={len(self.data['context']) + 1}")
        logger.debug(f"Chat input: {input_data}")
        
        try:
            loop = asyncio.get_event_loop()
            logger.debug("Using existing event loop")
        except RuntimeError:
            logger.debug("Creating new event loop")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        full_context = self._prepare_full_context(input_data, context)
        
        return loop.run_until_complete(
            send_request(full_context)
        )
    
    def _prepare_full_context(self, input_data: dict, context: list[dict],):
        full_context = context + [input_data]
        logger.debug(f"Full context prepared | total_len={len(full_context)}")
        return full_context
    
    # --- Utils ---
    # -- Update Data --
    def _update_data(self, data: dict):
        try:
            try:
                update_data(data)
                logger.debug(f"Current data updated successfully")
            except FileNotFoundError:
                logger.error("✗ Current data file not found error occurred while updating data")

            try:
                last_context = data['context'][-1]
                add_dialogue_entry(
                    user=last_context.get('/user', ''),
                    sys=last_context.get('/sys', ''),
                    cv=last_context.get('/cv', ''),
                    model=last_context.get('/model', ''),
                    hint=last_context.get('/hint', '')
                )
                logger.debug(f"Dialog data updated successfully")
            except FileNotFoundError:
                logger.error("✗ Dialog file not found error occurred while updating data")
        except Exception as e:
            logger.error(f"✗ Error occurred while updating data: {type(e).__name__}: {e}", exc_info=True)







if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("Starting Core module standalone test")
    logger.info("=" * 50)
    
    core = Core()
    
    test_text = "test stt input"
    logger.info(f"Sending test input: '{test_text}'")
    core.add_stt_input(test_text, "Test Speaker", "en")
    logger.info("Test completed")
    logger.info("=" * 50)