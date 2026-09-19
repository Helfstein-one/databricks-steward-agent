from src.domain.ports.databricks_port import IDatabricksAdapter


class SchemaIntrospectionUseCase:
    def __init__(self, db_adapter: IDatabricksAdapter):
        self.db = db_adapter

    def execute(self) -> str:
        return self.db.inspect_schema()
