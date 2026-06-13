"""
User entity - stores registered user accounts.
"""

from sqlalchemy import Integer, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class User(Base):
    __tablename__ = "users"
    __table_args__ = {"sqlite_autoincrement": True}

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    username: Mapped[str] = mapped_column(
        Text,
        unique=True,
        nullable=False,
    )
    email: Mapped[str] = mapped_column(
        Text,
        unique=True,
        nullable=False,
    )
    password_hash: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    created_at: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("(datetime('now'))"),
    )
    last_login_at: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    is_active: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("1"),
    )

    sessions = relationship(
        "ChatHistory",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<User {self.username} ({self.id})>"
