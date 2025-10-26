from abc import ABC, abstractmethod
from datetime import datetime

# --- Infra de auth/auditoría (mocks) ---
class Authz:
    @staticmethod
    def require(permission: str, user: str):
        if user != "admin" and permission != "READ_DOC":
            raise PermissionError(f"User '{user}' lacks '{permission}'")

class Audit:
    @staticmethod
    def log(event: str, user: str, resource: str):
        print(f"{datetime.utcnow().isoformat()} | {user} | {event} | {resource}")

# --- Repositorio base ---
class Document(dict): ...

class DocumentRepository(ABC):
    @abstractmethod
    def get(self, doc_id: str) -> Document: ...
    @abstractmethod
    def save(self, doc: Document) -> None: ...
    @abstractmethod
    def delete(self, doc_id: str) -> None: ...

class InMemoryDocumentRepository(DocumentRepository):
    def __init__(self):
        self._db: dict[str, Document] = {}
    def get(self, doc_id: str) -> Document:
        return self._db.get(doc_id, Document(id=doc_id, text=None))
    def save(self, doc: Document) -> None:
        self._db[doc["id"]] = doc
    def delete(self, doc_id: str) -> None:
        self._db.pop(doc_id, None)

# --- Proxy seguro con auditoría ---
class SecuredRepoProxy(DocumentRepository):
    def __init__(self, target: DocumentRepository, current_user: str):
        self._target = target
        self._user = current_user
    def get(self, doc_id: str) -> Document:
        Authz.require("READ_DOC", self._user)
        doc = self._target.get(doc_id)
        Audit.log("READ", self._user, doc_id)
        return doc
    def save(self, doc: Document) -> None:
        Authz.require("WRITE_DOC", self._user)
        self._target.save(doc)
        Audit.log("WRITE", self._user, doc["id"])
    def delete(self, doc_id: str) -> None:
        Authz.require("DELETE_DOC", self._user)
        self._target.delete(doc_id)
        Audit.log("DELETE", self._user, doc_id)

# Uso:
repo_real = InMemoryDocumentRepository()
repo = SecuredRepoProxy(repo_real, current_user="admin")
repo.save(Document(id="doc-123", text="Contenido sensible"))
doc = repo.get("doc-123")
