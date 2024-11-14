from aaa.core_processor import CoreProcessor


class UserInterface:


    def update_document(self):
        proc = CoreProcessor(
            original_document=None,
            document_name=None,
            new_tables=[],
            new_images=[]
        )
        proc.update_document()
