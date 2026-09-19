from src.domain.ports.databricks_port import IDatabricksAdapter


class DataPreviewUseCase:
    def __init__(self, db_adapter: IDatabricksAdapter):
        self.db = db_adapter

    def execute(self, table_name: str, limit: int = 10) -> str:
        if not table_name:
            return "❌ Nenhuma tabela identificada para consulta."
        return self.db.preview_table(table_name, limit)
