import bcrypt
import MySQLdb

# Connect to the database
db = MySQLdb.connect(host="localhost", user="root", passwd="", db="flask_app_klasifikasklng")
cursor = db.cursor()

# Fetch all users with their passwords
cursor.execute("SELECT id, password FROM users")
users = cursor.fetchall()

# Iterate over each user and verify their password hash
for user in users:
    user_id = user[0]
    password_hash = user[1].encode('utf-8')

    # Check if the password hash is valid
    try:
        bcrypt.checkpw(b'test', password_hash)  # Use a dummy password to check
    except ValueError:
        # If the password hash is invalid, rehash it
        new_password_hash = bcrypt.hashpw(password_hash, bcrypt.gensalt())
        # Update the password hash in the database
        cursor.execute("UPDATE users SET password = %s WHERE id = %s", (new_password_hash, user_id))
        db.commit()

# Close the database connection
cursor.close()
db.close()

