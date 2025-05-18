from easyner.io.database.connection import DatabaseConnection


def test_is_in_transaction():
    """Test the is_in_transaction method."""
    db = DatabaseConnection()
    db.connect()
    print("Connection established")
    print("Is in transaction:", db.is_in_transaction())

    assert not db.is_in_transaction()
    db.begin_transaction()
    assert db.is_in_transaction()
    db._conn.rollback()
    db.close()
