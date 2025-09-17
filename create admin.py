# create_admin.py
import sqlite3
import hashlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

DB_PATH = "production_app.db"

# --- Password Hashing ---
def hash_password(password):
    salt = "static_salt_change_me"  # must match your app's salt
    return hashlib.sha256((salt + password).encode()).hexdigest()

# --- Email Notification with Brevo ---
def send_email(to_email, subject, body):
    sender_email = "hafsatuxy@gmail.com"   # must be verified in Brevo
    brevo_login = "96cf99001@smtp-brevo.com"  # from Brevo dashboard
    brevo_password = "OKnmRy6V7fUY509I"       # SMTP key from Brevo
    smtp_server = "smtp-relay.brevo.com"
    smtp_port = 587

    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(brevo_login, brevo_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()

        print("✅ Email sent successfully!")
        return True
    except Exception as e:
        print("❌ Failed to send email:", e)
        return False

# --- Create Admin Function ---
def create_admin(username, password, email):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    hashed_pw = hash_password(password)

    # Ensure no duplicate admin with same username
    cur.execute("DELETE FROM users WHERE username = ?", (username,))

    # Insert admin (correct column name: password_hash)
    # cur.execute("""
    #     INSERT INTO users (username, email, password_hash, is_admin, created_at)
    #     VALUES (?, ?, ?, ?, ?)
    # """, (username, email, hashed_pw, 1, datetime.utcnow().isoformat()))

    # Insert admin (correct column name: password_hash)
    cur.execute("""
       INSERT INTO users (username, password_hash, is_admin, email, is_verified, created_at)
    VALUES (?, ?, ?, ?, ?, datetime('now'))
    """, (username, hashed_pw, 1, email, 1))


    conn.commit()
    conn.close()

    print(f"✅ Admin '{username}' created successfully.")

    # Send notification email
    subject = "🔐 New Admin Account Created"
    body = f"Hello {username},\n\nYour admin account has been created successfully.\n\nEmail: {email}\nRole: Admin"
    send_email(email, subject, body)

# --- Run directly ---
if __name__ == "__main__":
    create_admin("superadmin", "StrongPass123!", "hafsatuxy@gmail.com")
