from data_processing import preprocess, seq_to_ids, ids_to_seq, make_padding_mask
from vocabulary import vocabulary


class Chat:

    cotext = ""

    def _prep_request_text(self,text:str):
        text_with_role = "<USE>" + text + "<EOS>"
        text_tokens = preprocess(text_with_role)
        text_ids = seq_to_ids(text_tokens, vocabulary)

        return text_ids
    
    def _prep_rag_info_text(self, text:str):
        rag_with_role = "<SYS>" + text + "<EOS>"
        rag_info_tokens = preprocess(rag_with_role)
        rag_info_ids = seq_to_ids(rag_info_tokens, vocabulary)

        return rag_info_ids
    
    def get_full_request(self,request: str, rag_info: str = None):
        request = self._prep_request_text(request)
        rag_info = self._prep_rag_info_text(rag_info)

        full_reqest = self.cotext + rag_info + request

        self.cotext += request

        return full_reqest
    

    def get_answer_text(model_answer: list):
        #  logic to delete comads from responce and form a text to tts  model
        pass

    def get_commands_list(modle_answer: list):
        #  logic to delete text from responce and for list of commans for castome console
        pass



