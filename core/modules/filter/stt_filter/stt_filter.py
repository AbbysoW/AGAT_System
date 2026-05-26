import logging
from .imp_model import SttImpModel
from .should_shut_up import ShouldShutUpModel

logger = logging.getLogger(__name__)

class SttFilter:
    dialogue_continues = False

    def __init__(self):
        logger.info("Initializing SttFilter...")
        try:
            logger.debug("Loading SttImpModel...")
            self.imp_model = SttImpModel()
            logger.debug("Loading ShouldShutUpModel...")
            self.ssu_model = ShouldShutUpModel()
            logger.debug("✓ Models loaded")
            logger.info("✓ SttFilter initialized successfully")
        except Exception as e:
            logger.error(f"✗ SttFilter init error: {type(e).__name__}: {e}", exc_info=True)
            raise


    def __call__(self, input: dict, context: list[dict]) -> bool:
        logger.debug(f"SttFilter called | input_text='{input.get('data', {}).get('text', 'N/A')[:50]}...' | dialogue_continues={self.dialogue_continues}")
        
        try:

            threshold = self.get_threshold()
            logger.debug(f"Importance threshold: {threshold} (dialogue_continues={self.dialogue_continues})")
            
            is_important = self.is_important(input.get('data', {}), context, threshold)
            logger.debug(f"Importance: {is_important} (threshold: {threshold})")
            if not is_important:
                self.set_dialogue_continues(False)
                return False


            self.set_dialogue_continues(True)
            return True
        except Exception as e:
            logger.error(f"✗ Error in SttFilter: {type(e).__name__}: {e}", exc_info=True)
            return False


    

    
    
    
    
    # --- Is Important --
    def is_important(self, input: dict, context: list[dict], threshold: float) -> bool:
        """Check if input is important based on dialogue state and score."""
        logger.debug(f"Checking importance | threshold={threshold} | context_size={len(context)}")
        
        if not self.dialogue_continues:
            logger.debug("Dialogue not active - using empty context for evaluation")
            context = []
        else:
            logger.debug(f"Dialogue active - using context of size {len(context)}")
        
        imp = self._get_importance(input, context)
        logger.debug(f"Importance value: {imp:.3f} | Threshold: {threshold} | Important: {imp > threshold}")
        
        return imp > threshold
        
    def _get_importance(self, input: dict, context: list[dict]) -> float:
        """Get importance score from the model."""
        logger.debug("Getting importance score from SttImpModel...")
        try:
            score = self.imp_model(input, context)
            logger.debug(f"✓ Importance score obtained: {score:.3f}")
            return score
        except Exception as e:
            logger.error(f"✗ Error getting importance score: {type(e).__name__}: {e}", exc_info=True)
            return 0.0
        

    # --- Should Shut Up ---
    def should_shut_up(self, input: dict) -> bool:
        logger.debug(f"Checking if should shut up | input_text='{input.get('text', 'N/A')[:50]}...'")
        if self._should_shut_up(input) > 0.5:
            logger.info("🔇 Should shut up condition detected")
            self.set_dialogue_continues(False)
            return True
        return False

    def _should_shut_up(self, input: dict) -> float:
        """Determine if the system should remain silent."""
        logger.debug("Checking if should shut up...")
        should_shut = self.ssu_model(input)
        logger.debug(f"Should shut up result: {should_shut}")
        return should_shut
    

    # --- Utils ---
    def get_threshold(self) -> float:
        """Get the importance threshold based on dialogue state."""
        if self.dialogue_continues:
            threshold = 0.5
            logger.debug(f"Dialogue active - using lower threshold: {threshold}")
        else:
            threshold = 0.8
            logger.debug(f"Dialogue not active - using higher threshold: {threshold}")
        return threshold
    
    def set_dialogue_continues(self, continues: bool):
        """Set the dialogue state."""
        old_state = self.dialogue_continues
        self.dialogue_continues = continues
        if old_state != continues:
            logger.info(f"Dialogue state changed: {old_state} → {continues}")
        else:
            logger.debug(f"Dialogue state unchanged: {continues}")
