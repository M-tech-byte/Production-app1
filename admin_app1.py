# #####################################################################
# # production_manager_full.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import sqlite3
# import os
# import hashlib
# import time
# from datetime import datetime, timedelta
# from io import BytesIO
# import matplotlib.pyplot as plt


# # optional: KNN
# try:
#     from sklearn.neighbors import KNeighborsRegressor
#     SKLEARN_AVAILABLE = True
# except Exception:
#     SKLEARN_AVAILABLE = False

# # ------------------ CONFIG ------------------
# DB_PATH = "production_app.db"
# DATA_DIR = "data_uploads"
# os.makedirs(DATA_DIR, exist_ok=True)

# # ------------------ DB ------------------
# conn = sqlite3.connect(DB_PATH, check_same_thread=False)
# cur = conn.cursor()


# # # ------------------ DB ------------------
# # conn = sqlite3.connect(DB_PATH, check_same_thread=False)
# # cur = conn.cursor()


# # import sqlite3

# # DB_PATH = "production_app.db"

# # def reset_users_table():
# #     # Use a separate connection just for this operation
# #     with sqlite3.connect(DB_PATH) as conn:
# #         cur = conn.cursor()

# #         # Drop the table if it exists
# #         cur.execute("DROP TABLE IF EXISTS users")

# #         # Recreate the table
# #         cur.execute('''
# #         CREATE TABLE users (
# #             id INTEGER PRIMARY KEY AUTOINCREMENT,
# #             username TEXT UNIQUE,
# #             email TEXT UNIQUE,
# #             password_hash TEXT,
# #             is_admin INTEGER DEFAULT 0,
# #             is_verified INTEGER DEFAULT 0,
# #             verify_token TEXT,
# #             reset_token TEXT,
# #             created_at TEXT
# #         )
# #         ''')

# #         conn.commit()
# #         print("✅ Users table has been reset successfully!")

# # # Call this function once when you want to reset
# # reset_users_table()

# # --- Create users table if it doesn't exist ---
# cur.execute('''
# CREATE TABLE IF NOT EXISTS users (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     username TEXT UNIQUE,
#     email TEXT UNIQUE,
#     password_hash TEXT,
#     is_admin INTEGER DEFAULT 0,
#     created_at TEXT
# )
# ''')

# # --- Create uploads table if it doesn't exist ---
# cur.execute('''
# CREATE TABLE IF NOT EXISTS uploads (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     user_id INTEGER,
#     filename TEXT,
#     filepath TEXT,
#     uploaded_at TEXT,
#     field_name TEXT,
#     notes TEXT,
#     source TEXT DEFAULT 'upload',
#     FOREIGN KEY(user_id) REFERENCES users(id)
# )
# ''')

# conn.commit()

# # --- Ensure the users table has the additional columns ---
# def ensure_user_columns():
#     cur.execute("PRAGMA table_info(users)")
#     existing = [row[1] for row in cur.fetchall()]

#     if "is_verified" not in existing:
#         cur.execute("ALTER TABLE users ADD COLUMN is_verified INTEGER DEFAULT 0")
#     if "verify_token" not in existing:
#         cur.execute("ALTER TABLE users ADD COLUMN verify_token TEXT")
#     if "reset_token" not in existing:
#         cur.execute("ALTER TABLE users ADD COLUMN reset_token TEXT")
#     conn.commit()

# ensure_user_columns()

# # ------------------ HELPERS & NEW SCHEDULING LOGIC ------------------
# REQUIRED_COLS = ["Date", "Oil (BOPD)", "Gas (MMSCFD)"]

# def hash_password(password: str) -> str:
#     salt = "static_salt_change_me"  # replace in production
#     return hashlib.sha256((salt + password).encode()).hexdigest()

# def create_user(username, email, password, is_admin=0):
#     ph = hash_password(password)
#     try:
#         cur.execute("INSERT INTO users (username,email,password_hash,is_admin,created_at) VALUES (?,?,?,?,?)",
#                     (username, email, ph, int(is_admin), datetime.utcnow().isoformat()))
#         conn.commit()
#         return True, "user created"
#     except sqlite3.IntegrityError as e:
#         return False, str(e)


# def authenticate(username_or_email, password):
#     ph = hash_password(password)
#     cur.execute(
#         "SELECT id, username, email, is_admin, is_verified FROM users WHERE (username=? OR email=?) AND password_hash=?",
#         (username_or_email, username_or_email, ph)
#     )
#     row = cur.fetchone()


#     cur.execute("SELECT id, username, email, is_verified FROM users")
#     print(cur.fetchall())

#     if row:
#         user = {
#             "id": row[0],
#             "username": row[1],
#             "email": row[2],
#             "is_admin": row[3],
#             "is_verified": row[4]  # ✅ include this!
#         }
#         return True, user
#     else:
#         return False, None

# def validate_and_normalize_df(df: pd.DataFrame):
#     # Normalize column names and check required
#     colmap = {}
#     for c in df.columns:
#         c_clean = c.strip()
#         if c_clean.lower() in [x.lower() for x in REQUIRED_COLS]:
#             for req in REQUIRED_COLS:
#                 if c_clean.lower() == req.lower():
#                     colmap[c] = req
#         elif 'date' in c_clean.lower():
#             colmap[c] = 'Date'
#         elif 'oil' in c_clean.lower():
#             colmap[c] = 'Oil (BOPD)'
#         elif 'gas' in c_clean.lower():
#             colmap[c] = 'Gas (MMSCFD)'
#         else:
#             colmap[c] = c_clean
#     df = df.rename(columns=colmap)
#     missing = [c for c in REQUIRED_COLS if c not in df.columns]
#     if missing:
#         return False, f"Missing required columns: {missing}"
#     try:
#         df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
#     except Exception as e:
#         return False, f"Date parse error: {e}"
#     if df['Date'].isna().any():
#         return False, "Some Date values could not be parsed. Ensure ISO-like dates (YYYY-MM-DD)."
#     for c in ['Oil (BOPD)','Gas (MMSCFD)']:
#         df[c] = pd.to_numeric(df[c], errors='coerce')
#         if df[c].isna().all():
#             return False, f"Column {c} has no numeric values"
#     return True, df

# def save_dataframe(user_id, df: pd.DataFrame, field_name: str, notes: str = "", source: str = "upload"):
#     ts = int(time.time())
#     safe_field = field_name.replace(" ", "_")[:60] if field_name else "unnamed"
#     filename = f"user{user_id}_{safe_field}_{ts}.csv"
#     filepath = os.path.join(DATA_DIR, filename)
#     df.to_csv(filepath, index=False)
#     cur.execute("INSERT INTO uploads (user_id,filename,filepath,uploaded_at,field_name,notes,source) VALUES (?,?,?,?,?,?,?)",
#                 (user_id, filename, filepath, datetime.utcnow().isoformat(), field_name, notes, source))
#     conn.commit()
#     return filename, filepath

# def apply_deferments(df, deferments):
#     """Apply deferments by zeroing Oil & Gas in deferment ranges."""
#     if df is None or df.empty or not deferments:
#         return df
#     df = df.copy()
#     if 'Date' not in df.columns:
#         return df
#     df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
#     for d in deferments.values():
#         reason = d.get('reason')
#         days = int(d.get('duration_days', 0) or 0)
#         start_str = d.get('start_date')
#         if reason and reason != "None" and days > 0 and start_str:
#             try:
#                 start = pd.to_datetime(start_str, errors='coerce')
#                 if pd.isna(start):
#                     continue
#                 end = start + pd.Timedelta(days=days-1)
#                 mask = (df['Date'] >= start) & (df['Date'] <= end)
#                 if 'Oil (BOPD)' in df.columns:
#                     df.loc[mask, 'Oil (BOPD)'] = 0
#                 if 'Gas (MMSCFD)' in df.columns:
#                     df.loc[mask, 'Gas (MMSCFD)'] = 0
#                 if 'Production' in df.columns:
#                     df.loc[mask, 'Production'] = 0
#             except Exception:
#                 continue
#     return df

# def shade_deferment_spans(ax, deferments, color_map=None):
#     spans = []
#     for d in deferments.values():
#         reason = d.get('reason')
#         days = int(d.get('duration_days', 0) or 0)
#         start_str = d.get('start_date')
#         if reason and reason != "None" and days > 0 and start_str:
#             s = pd.to_datetime(start_str, errors='coerce')
#             if pd.isna(s):
#                 continue
#             e = s + pd.Timedelta(days=days-1)
#             spans.append((s, e, reason))
#     for s, e, r in spans:
#         color = None
#         if color_map and r in color_map:
#             color = color_map[r]
#         ax.axvspan(s, e, alpha=0.18, color=color)

# # ------------------ NEW: SCHEDULING & TARGET AVERAGE HELPERS ------------------

# def _ensure_date_index(df):
#     """Return a copy indexed by Date (datetime), preserving original Date column too."""
#     d = df.copy()
#     if 'Date' in d.columns:
#         d['Date'] = pd.to_datetime(d['Date'], errors='coerce')
#         d = d.sort_values('Date').reset_index(drop=True)
#         d_indexed = d.set_index('Date', drop=False)
#     else:
#         # Try to use existing index
#         idx = pd.to_datetime(d.index)
#         d_indexed = d.copy()
#         d_indexed['Date'] = idx
#         d_indexed.index = idx
#     return d_indexed

# def apply_scheduled_changes(df, schedules, column):
#     """
#     Apply scheduled ramp-ups/declines with linear interpolation on the specified column.
#     schedules: dict where values are dicts (as stored in session_state) OR a mapping date->target
#     Accepts input df that has a 'Date' column or datetime index. Returns df with same shape.
#     """
#     if df is None or df.empty or not schedules or column not in df.columns:
#         return df

#     # Convert to DataFrame indexed by Date
#     df_i = _ensure_date_index(df)

#     # Accept two possible schedule formats:
#     # 1) state dict as in session_state: {'ramp_1': {'date':'YYYY-MM-DD','rate':5000}, ...}
#     # 2) direct mapping: {'2025-09-01': 5000, ...}
#     mapping = {}
#     # if schedules is a mapping of date->value (string/number)
#     if all((isinstance(k, (str, pd.Timestamp)) and not isinstance(v, dict)) for k, v in schedules.items()):
#         # treat as direct mapping
#         for k, v in schedules.items():
#             try:
#                 d = pd.to_datetime(k)
#                 mapping[d] = float(v)
#             except Exception:
#                 continue
#     else:
#         # treat as state dict
#         for entry in schedules.values():
#             try:
#                 d = pd.to_datetime(entry.get('date'))
#             except Exception:
#                 continue
#             if pd.isna(d):
#                 continue
#             # If entry has 'rate' it's absolute target
#             if 'rate' in entry:
#                 try:
#                     mapping[d] = float(entry.get('rate'))
#                 except Exception:
#                     continue
#             # If entry has 'pct' treat as a percent change to the base value at that date
#             elif 'pct' in entry:
#                 try:
#                     pct = float(entry.get('pct')) / 100.0
#                 except Exception:
#                     pct = 0.0
#                 # find base value just before or at that date
#                 if d in df_i.index:
#                     base = df_i.loc[d, column]
#                     if pd.isna(base):
#                         # take previous non-na
#                         prev = df_i.loc[:d, column].ffill().iloc[-1] if not df_i.loc[:d, column].ffill().empty else None
#                         base = prev if prev is not None else 0.0
#                 else:
#                     # take last known before date
#                     prev_series = df_i.loc[df_i.index <= d, column]
#                     if not prev_series.empty:
#                         base = prev_series.ffill().iloc[-1]
#                     else:
#                         # fallback: use earliest value
#                         base = df_i[column].ffill().iloc[0] if not df_i[column].ffill().empty else 0.0
#                 mapping[d] = float(base * (1 - pct))

#     if not mapping:
#         return df

#     # Keep only mapping dates within the df index range (we can still insert future dates if df extends into forecast)
#     # Insert scheduled targets (create rows if necessary)
#     for d, target in mapping.items():
#         if d not in df_i.index:
#             # if date outside range, add a new row so interpolation can happen
#             # create a new row using NaN for all columns, set Date index to d
#             new_row = pd.DataFrame([{c: np.nan for c in df_i.columns}])
#             new_row.index = pd.Index([d])
#             new_row['Date'] = d
#             df_i = pd.concat([df_i, new_row])
#     # Re-sort index
#     df_i = df_i.sort_index()

#     # Set scheduled absolute values
#     for d, target in mapping.items():
#         df_i.at[d, column] = target

#     # Interpolate the column across the entire index (linear)
#     df_i[column] = df_i[column].interpolate(method='time')  # time-aware interpolation

#     # Return df_i with original order and with Date column preserved
#     # Remove any rows we artificially added that were outside original df's full date coverage only if original had no such dates
#     # To be safe, convert back to same index shape as original input by reindexing to original index if original had explicit index
#     out = df_i.reset_index(drop=True)
#     # Recreate the original shape: ensure it has same number of rows as before if original index used; but we often want the scheduled rows kept
#     # We'll return df_i.reset_index(drop=True) but keep Date column (so other code that groups by Date still works)
#     return df_i.reset_index(drop=True).copy()

# def apply_target_average(df, target_avg=None, column=None, lock_schedules=False):
#     """
#     Scale a column in df to match target_avg.
#     If lock_schedules=True, it will try to preserve points where values were explicitly set (i.e., large jumps).
#     Note: target_avg expected to be average over the dataframe (not per-day).
#     """
#     if df is None or df.empty or column not in df.columns or target_avg is None:
#         return df
#     d = df.copy()
#     cur_avg = d[column].mean()
#     if cur_avg <= 0:
#         return d
#     scale = float(target_avg) / float(cur_avg)
#     if not lock_schedules:
#         d[column] = d[column] * scale
#     else:
#         # Attempt to detect scheduled points as places with sudden changes (non-small diffs)
#         diffs = d[column].diff().abs()
#         # threshold: use median*5 or minimal fallback
#         thresh = max(diffs.median() * 5.0, 1e-6)
#         scheduled_idx = diffs[diffs > thresh].index.tolist()
#         # Scale only indices that are not scheduled
#         mask = ~d.index.isin(scheduled_idx)
#         d.loc[mask, column] = d.loc[mask, column] * scale
#     return d

# # ------------------ UI SETUP ------------------
# st.set_page_config(page_title="Production Manager", layout="wide")
# if 'auth' not in st.session_state:
#     st.session_state['auth'] = {'logged_in': False}

# #########################################  LOGIN SECTION  #####################################
# import smtplib
# import hashlib, time, os
# from email.mime.text import MIMEText
# import hashlib, time, os
# from datetime import datetime, timedelta
# from email.mime.multipart import MIMEMultipart
# import streamlit as st





# def send_email(to_email, subject, body):
#     sender_email = "hafsatuxy@gmail.com"   # must be verified in Brevo
#     password = "OKnmRy6V7fUY509I"
#     smtp_server = "smtp-relay.brevo.com"
#     smtp_port = 587

#     try:
#         msg = MIMEMultipart()
#         msg["From"] = sender_email
#         msg["To"] = to_email
#         msg["Subject"] = subject
#         msg.attach(MIMEText(body, "plain"))

#         server = smtplib.SMTP(smtp_server, smtp_port)
#         server.starttls()
#         server.login("96cf99001@smtp-brevo.com", password)  # login always with Brevo login
#         server.sendmail(sender_email, to_email, msg.as_string())
#         server.quit()

#         print("✅ Email sent successfully!")
#         return True
#     except Exception as e:
#         print("❌ Failed to send email:", e)
#         return False

# # ------------------ SIDEBAR: AUTH, DURATION ------------------
# st.sidebar.header("Account")

# # action = st.sidebar.selectbox("Action", ["Login", "Sign up", "Forgot password", "Verify account"])
# action = st.sidebar.selectbox(
#     "Action",
#     ["Login", "Sign up", "Forgot password", "Verify account", "Delete account"]
# )


# # ---------- SIGN UP ----------
# # Ensure user is always defined to avoid NameError
# user = st.session_state.get('auth', {}).get('user', None)





# def create_user(username, email, password, is_admin=0, is_verified=0):
#     password_hash = hash_password(password)
#     try:
#         cur.execute(
#             """
#             INSERT INTO users (username, email, password_hash, is_admin, is_verified, verify_token, reset_token)
#             VALUES (?, ?, ?, ?, ?, NULL, NULL)
#             """,
#             (username, email, password_hash, is_admin, is_verified)
#         )
#         conn.commit()
#         return True, "User created successfully"
#     except Exception as e:
#         return False, str(e)







# if action == 'Sign up':
#     st.sidebar.markdown("Create a new account")
#     su_user = st.sidebar.text_input("Username", key='su_user')
#     su_email = st.sidebar.text_input("Email", key='su_email')
#     su_pass = st.sidebar.text_input("Password", type='password', key='su_pass')
#     su_pass2 = st.sidebar.text_input("Confirm password", type='password', key='su_pass2')
    
#     if st.sidebar.button("Create account", key="create_account_btn"):
#         if su_pass != su_pass2:
#             st.sidebar.error("Passwords do not match")
#         elif not su_user or not su_pass or not su_email:
#             st.sidebar.error("Fill all fields")
#         else:
#             # 🚫 Force all signups as normal users
#             is_admin_flag = 0  

#             ok, msg = create_user(
#                 su_user, su_email, su_pass,
#                 is_admin=is_admin_flag,
#                 is_verified=0
#             )

#             if ok:
#                 # Generate verification token
#                 token = hashlib.sha1((su_email + str(time.time())).encode()).hexdigest()[:8]
#                 cur.execute("UPDATE users SET verify_token=? WHERE email=?", (token, su_email))
#                 conn.commit()

#                 # Send email with token
#                 send_email(
#                     su_email,
#                     "Verify Your Account - Production App",
#                     f"Hello {su_user},\n\nThank you for signing up.\n\n"
#                     f"Your verification code is: {token}\n\n"
#                     f"Enter this code in the app to activate your account."
#                 )
#                 st.success(f"✅ Signup successful! A verification email was sent to {su_email}. Please check your inbox (or spam).")
#             else:
#                 st.sidebar.error(msg)

# # ---------- VERIFY ACCOUNT ----------
# elif action == "Verify account":
#     st.sidebar.markdown("Enter the code sent to your email")
#     ver_email = st.sidebar.text_input("Email", key="ver_email")
#     ver_code = st.sidebar.text_input("Verification code", key="ver_code")

#     if st.sidebar.button("Verify now", key="verify_btn"):
#         cur.execute("SELECT id, verify_token FROM users WHERE email=?", (ver_email,))
#         r = cur.fetchone()
#         if r and r[1] == ver_code:
#             cur.execute("UPDATE users SET is_verified=1, verify_token=NULL WHERE id=?", (r[0],))
#             conn.commit()
#             st.sidebar.success("✅ Account verified. You can now login.")
#         else:
#             st.sidebar.error("❌ Invalid verification code")

#     # 🔹 Send verification code
#     if st.sidebar.button("Send Verification Code", key="resend_btn"):
#         cur.execute("SELECT verify_token FROM users WHERE email=?", (ver_email,))
#         r = cur.fetchone()

#         if r:
#             code = r[0]
#             subject = "Your Verification Code"
#             body = f"Hello,\n\nYour verification code is: {code}\n\nUse this to verify your account."
#             try:
#                 send_email(ver_email, subject, body)
#                 st.sidebar.success("📩 Verification code sent to your email!")
#             except Exception as e:
#                 st.sidebar.error(f"❌ Failed to send code: {e}")
#         else:
#             st.sidebar.warning("⚠️ Email not found. Please sign up first.")

# # ---------- FORGOT PASSWORD ----------
# elif action == 'Forgot password':
#     st.sidebar.markdown("Reset password")
#     fp_email = st.sidebar.text_input("Enter your email", key='fp_email')

#     if st.sidebar.button("Send reset code", key="reset_code_btn"):
#         cur.execute("SELECT id FROM users WHERE email=?", (fp_email,))
#         r = cur.fetchone()
#         if r:
#             token = hashlib.sha1((fp_email + str(time.time())).encode()).hexdigest()[:8]

#             cur.execute("UPDATE users SET reset_token=? WHERE id=?", (token, r[0]))
#             conn.commit()

#             send_email(
#                 fp_email,
#                 "Password Reset - Production App",
#                 f"Your reset code is: {token}\n\n"
#                 "Use this code in the app to set a new password."
#             )
#             st.sidebar.success("A reset code has been sent to your email.")
#         else:
#             st.sidebar.error("Email not found")

#     st.sidebar.markdown("---")
#     token_user = st.sidebar.text_input("User ID (from your account)", key="token_user")
#     token_val = st.sidebar.text_input("Reset code", key="token_val")
#     token_newpw = st.sidebar.text_input("New password", type='password', key="token_newpw")

#     if st.sidebar.button("Reset password now", key="reset_now_btn"):
#         try:
#             uid = int(token_user)
#             cur.execute("SELECT reset_token FROM users WHERE id=?", (uid,))
#             db_token = cur.fetchone()
#             if db_token and db_token[0] == token_val:
#                 ph = hash_password(token_newpw)
#                 cur.execute("UPDATE users SET password_hash=?, reset_token=NULL WHERE id=?", (ph, uid))
#                 conn.commit()
#                 st.sidebar.success("Password updated. Please login.")
#             else:
#                 st.sidebar.error("Invalid or expired token")
#         except Exception:
#             st.sidebar.error("Invalid user id")

# # ---------- LOGIN ----------
# elif action == "Login":
#     login_user = st.sidebar.text_input("Username or email", key='login_user')
#     login_pass = st.sidebar.text_input("Password", type='password', key='login_pass')

#     dur_map = {
#         "1 minute": 60, "5 minutes": 300, "30 minutes": 1800,
#         "1 hour": 3600, "1 day": 86400, "7 days": 604800, "30 days": 2592000
#     }
#     sel_dur = st.sidebar.selectbox("Login duration", list(dur_map.keys()), index=3)

#     if st.sidebar.button("Login", key="login_btn"):
#         ok, user = authenticate(login_user, login_pass)
#         if ok:
#             # ✅ Ensure verified check works properly
#             if not user.get('is_verified') or user['is_verified'] == 0:
#                 st.sidebar.error("⚠️ Please verify your email before logging in.")
#             else:
#                 st.session_state['auth'] = {
#                     'logged_in': True,
#                     'user': user,
#                     'expires_at': (datetime.utcnow() + timedelta(seconds=dur_map[sel_dur])).timestamp()
#                 }
#                 # st.sidebar.success(f"✅ Logged in as {user['username']}")

#                 # Send login notification email
#                 send_email(
#                     user['email'],
#                     "Login Notification - Production App",
#                     f"Hello {user['username']},\n\n"
#                     f"Your account was just logged in on {datetime.utcnow()} (UTC)."
#                 )
#         else:
#             st.sidebar.error("❌ Invalid credentials. If you are a new user, please sign up.")

#     # --- After login/signup, check session ---
#     auth = st.session_state.get("auth", {})
#     user = auth.get("user")

#     if not auth.get("logged_in"):
#         # Not logged in yet
#         st.title("Welcome — To the Production App")
#         st.write("Please sign up or login from the sidebar.")
#         st.stop()  # 🔴 Prevents the rest of the app from running
#     else:
#         # ✅ Logged in successfully → safe to use user['id']
#         st.sidebar.success(f"Logged in as {user['username']}")

#         # Example: show uploads for this user
#         cur.execute(
#             "SELECT id, filename, filepath, field_name, uploaded_at FROM uploads WHERE user_id=? ORDER BY uploaded_at DESC",
#             (user['id'],)
#         )
#         uploads = cur.fetchall()

#         st.write("### Your Uploads")
#         if uploads:
#             for up in uploads:
#                 st.write(f"📂 {up[1]} (uploaded {up[4]})")
#         else:
#             st.write("No uploads yet.")

#         if st.sidebar.button("Logout", key="logout_btn"):
#             st.session_state["auth"] = {"logged_in": False, "user": None}
#             st.experimental_rerun()

# elif action == "Delete account":
#     st.sidebar.markdown("Delete your account permanently")

#     del_user = st.sidebar.text_input("Username or Email", key="del_user")
#     del_pass = st.sidebar.text_input("Password", type="password", key="del_pass")

#     if st.sidebar.button("Delete my account", key="del_btn"):
#         ok, user = authenticate(del_user, del_pass)
#         if ok and user:
#             # Delete account from DB
#             cur.execute("DELETE FROM users WHERE id=?", (user['id'],))
#             conn.commit()

#             # Clear session
#             if "auth" in st.session_state:
#                 st.session_state.pop("auth")

#             st.sidebar.success("Your account has been permanently deleted.")
#             st.session_state["user"] = None

#             # Optional: send email notification
#             send_email(
#                 del_user,
#                 "Account Deleted - Production App",
#                 f"Hello,\n\nYour account ({del_user}) has been permanently deleted."
#             )
#         else:
#             st.sidebar.error("Invalid username/email or password")

# # ------------------ SIDEBAR: Deferments (100) ------------------
# with st.sidebar.expander("⚙️ Deferment Controls", expanded=False):
#     if "deferments" not in st.session_state:
#         st.session_state["deferments"] = {}

#     options = ["None", "Maintenance", "Pipeline Issue", "Reservoir", "Other"]
#     for i in range(1, 6):  # reduced to 5 in UI for practicality; original allowed 100. adjust as needed
#         st.markdown(f"**Deferment {i}**")
#         reason = st.selectbox(f"Reason {i}", options, key=f"deferment_reason_{i}")
#         start_date = st.date_input(f"Start Date {i}", key=f"deferment_start_{i}")
#         duration = st.number_input(f"Duration (days) {i}", min_value=0, max_value=3650, step=1, key=f"deferment_duration_{i}")
#         st.session_state["deferments"][f"Deferment {i}"] = {
#             "reason": reason,
#             "start_date": str(start_date),
#             "duration_days": int(duration)
#         }

#     st.markdown("### 📋 Active Deferments")
#     active = {k:v for k,v in st.session_state["deferments"].items() if v["reason"] != "None" and v["duration_days"] > 0}
#     if active:
#         for name, d in active.items():
#             st.markdown(f"- **{name}** → {d['reason']} starting {d['start_date']} for {d['duration_days']} days")
#     else:
#         st.info("No active deferments set")

# # ===========================
# # SIDEBAR CONTROLS - RAMP & DECLINE
# # ===========================
# with st.sidebar.expander("⚡ Ramp-up & Decline Controls", expanded=False):
#     if "rampups" not in st.session_state:
#         st.session_state["rampups"] = {}
#     if "declines" not in st.session_state:
#         st.session_state["declines"] = {}
#     # input for a single ramp or decline then add to session_state (keeps original UI)
#     ramp_date = st.date_input("Ramp-up Start Date", key="ramp_date")
#     ramp_rate = st.number_input("New Ramp-up Rate (absolute)", min_value=0.0, value=5000.0, key="ramp_rate")
#     if st.button("Apply Ramp-up"):
#         st.session_state["rampups"][f"ramp_{len(st.session_state['rampups'])+1}"] = {
#             "date": str(ramp_date),
#             "rate": float(ramp_rate)
#         }
#         st.success(f"Ramp-up set: {ramp_date} → {ramp_rate}")

#     st.markdown("---")
#     decline_date = st.date_input("Decline Start Date", key="decline_date")
#     decline_pct = st.slider("Decline Percentage (%)", 0, 100, 10, key="decline_pct")
#     if st.button("Apply Decline"):
#         st.session_state["declines"][f"decline_{len(st.session_state['declines'])+1}"] = {
#             "date": str(decline_date),
#             "pct": int(decline_pct)
#         }
#         st.success(f"Decline set: {decline_date} → -{decline_pct}%")

#     st.markdown("---")
#     st.write("Active Ramps")
#     for k, v in st.session_state.get("rampups", {}).items():
#         st.write(f"- {k}: {v.get('date')} → {v.get('rate')}")

#     st.write("Active Declines")
#     for k, v in st.session_state.get("declines", {}).items():
#         st.write(f"- {k}: {v.get('date')} → -{v.get('pct')}%")

#     st.markdown("---")
#     st.write
#     ("Target avg (optional) — applied after scheduling")
#     if "target_avg" not in st.session_state:
#         st.session_state["target_avg"] = {"Oil": None, "Gas": None, "lock": True}
#     ta_oil = st.number_input("Target Avg Oil (BOPD) — leave 0 to ignore", min_value=0, value=int(st.session_state["target_avg"].get("Oil") or 0), key="ta_oil")
#     ta_gas = st.number_input("Target Avg Gas (MMSCFD) — leave 0 to ignore", min_value=0, value=int(st.session_state["target_avg"].get("Gas") or 0), key="ta_gas")
#     lock_sched = st.checkbox("Lock scheduled points (do not rescale them)", value=st.session_state["target_avg"].get("lock", True))
#     if st.button("Set Target Avg"):
#         st.session_state["target_avg"]["Oil"] = int(ta_oil) if ta_oil > 0 else None
#         st.session_state["target_avg"]["Gas"] = int(ta_gas) if ta_gas > 0 else None
#         st.session_state["target_avg"]["lock"] = bool(lock_sched)
#         st.success("Target averages stored in session")




# # --- After ALL auth actions (login/signup/verify/forgot/delete), check login state ---
# auth = st.session_state.get("auth", {})
# user = auth.get("user")

# if not auth.get("logged_in"):
#     # Not logged in yet
#     st.title("Welcome — To the Production App")
#     st.write("Please sign up or login from the sidebar.")
#     st.stop()  # 🔴 Prevents the rest of the app from running
# else:
#     # Logged in successfully → safe to use user['id']
#     # st.sidebar.success(f"Logged in as {user['username']}")

#     # Example: show uploads for this user
#     cur.execute(
#         "SELECT id, filename, filepath, field_name, uploaded_at FROM uploads WHERE user_id=? ORDER BY uploaded_at DESC",
#         (user['id'],)
#     )
#     uploads = cur.fetchall()

#     st.write("### Your Uploads")
#     if uploads:
#         for up in uploads:
#             st.write(f"📂 {up[1]} (uploaded {up[4]})")
#     else:
#         st.write("No uploads yet.")





# # ------------------ MAIN TABS ------------------
# st.title("Production App — Workspaces")
# tabs = st.tabs(["Production Input", "Forecast Analysis", "Account & Admin", "Recent Uploads", "Saved Files"])

# # ------------------ PRODUCTION INPUT ------------------
# with tabs[0]:
#     st.header("Production Input")
#     st.write("Upload CSV files or do manual entry. Required columns: Date, Oil (BOPD), Gas (MMSCFD).")

#     # Let user choose method
#     method = st.radio("Choose Input Method", ["Upload CSV", "Manual Entry"], horizontal=True)

#     if method == "Upload CSV":
#         uploaded = st.file_uploader("Upload CSV", type=['csv'], key='upl1')
#         field_name = st.text_input("Field name (e.g. OML 98)")
#         notes = st.text_area("Notes (optional)")
#         if st.button("Upload and validate"):
#             if uploaded is None:
#                 st.error("Please select a file to upload.")
#             else:
#                 try:
#                     df = pd.read_csv(uploaded)
#                 except Exception as e:
#                     st.error(f"Could not read CSV: {e}")
#                     df = None
#                 if df is not None:
#                     ok, out = validate_and_normalize_df(df)
#                     if not ok:
#                         st.error(out)
#                     else:
#                         df_clean = out
#                         ts = int(time.time())
#                         filename = f"{user['username']}_{field_name}_{ts}.csv"
#                         filepath = os.path.join(DATA_DIR, filename)
#                         df_clean.to_csv(filepath, index=False)
#                         cur.execute("INSERT INTO uploads (user_id,filename,filepath,uploaded_at,field_name,notes) VALUES (?,?,?,?,?,?)",
#                                     (user['id'], filename, filepath, datetime.utcnow().isoformat(), field_name, notes))
#                         conn.commit()
#                         st.success("Uploaded and saved")
#                         st.dataframe(df_clean.head())

#     else:  # Manual Entry
#         st.subheader("Manual Excel-like Workspace (Paste or Type)")

#         ws_name = st.text_input("Workspace name", key="manual_ws_name")

#         if "manual_table" not in st.session_state:
#             st.session_state["manual_table"] = pd.DataFrame({
#                 "Date": [""] * 8,
#                 "Oil (BOPD)": [None] * 8,
#                 "Gas (MMSCFD)": [None] * 8
#             })

#         st.info("You can paste large blocks (Date, Oil, Gas). Date format YYYY-MM-DD recommended.")
#         edited_df = st.data_editor(st.session_state["manual_table"], num_rows="dynamic", key="manual_table_editor", use_container_width=True)
#         st.session_state["manual_table"] = edited_df

#         if not edited_df.empty:
#             st.subheader(f"Preview — {ws_name if ws_name else 'Unnamed'}")
#             st.dataframe(edited_df.head(50))

#             totals = pd.DataFrame({
#                 "Date": ["TOTAL"],
#                 "Oil (BOPD)": [pd.to_numeric(edited_df["Oil (BOPD)"], errors="coerce").sum(skipna=True)],
#                 "Gas (MMSCFD)": [pd.to_numeric(edited_df["Gas (MMSCFD)"], errors="coerce").sum(skipna=True)]
#             })
#             st.write("**Totals:**")
#             st.dataframe(totals)

#             if st.button("Save Workspace to CSV"):
#                 if not ws_name.strip():
#                     st.error("Please enter a workspace name.")
#                 else:
#                     ok, norm = validate_and_normalize_df(edited_df)
#                     if not ok:
#                         st.error(norm)
#                     else:
#                         ts = int(time.time())
#                         filename = f"{user['username']}_{ws_name}_{ts}.csv"
#                         filepath = os.path.join(DATA_DIR, filename)
#                         norm.to_csv(filepath, index=False)
#                         cur.execute("INSERT INTO uploads (user_id,filename,filepath,uploaded_at,field_name,notes) VALUES (?,?,?,?,?,?)",
#                                     (user['id'], filename, filepath, datetime.utcnow().isoformat(), ws_name, "manual workspace"))
#                         conn.commit()
#                         st.success(f"Workspace saved as {filename}")

# # ------------------ FORECAST ANALYSIS ------------------
# with tabs[1]:
#     st.header("Forecast Analysis")
#     st.write("Combine uploaded files (or pick a file) and run analysis. Deferments will zero production in selected windows and be shaded on plots.")

#     # List user's uploads
#     cur.execute("SELECT id,filename,filepath,field_name,uploaded_at FROM uploads WHERE user_id=? ORDER BY uploaded_at DESC", (user['id'],))
#     myfiles = cur.fetchall()
#     files_df = pd.DataFrame(myfiles, columns=['id','filename','filepath','field_name','uploaded_at']) if myfiles else pd.DataFrame()

#     if files_df.empty:
#         st.info("No uploads found. Use Production Input to upload or create a manual workspace.")
#     else:
#         sel_files = st.multiselect("Select files to include in analysis (multiple allowed)", files_df['filename'].tolist())
#         if sel_files:
#             dfs = []
#             for fn in sel_files:
#                 fp = files_df.loc[files_df['filename'] == fn, 'filepath'].values[0]
#                 if os.path.exists(fp):
#                     df = pd.read_csv(fp, parse_dates=['Date'])
#                     df['source_file'] = fn
#                     dfs.append(df)
#             if not dfs:
#                 st.error("Selected files not found on disk.")
#             else:
#                 big = pd.concat(dfs, ignore_index=True)
#                 big['Date'] = pd.to_datetime(big['Date'], errors='coerce')
#                 st.subheader("Combined preview")
#                 st.dataframe(big.head(50))

#                 freq = st.selectbox("Aggregation freq", ['D', 'M', 'Y'], index=1, help='D=Daily, M=Monthly, Y=Yearly')
#                 horizon_choice = st.selectbox("Forecast horizon unit", ['Days', 'Months', 'Years'], index=2)
#                 horizon_value = st.number_input("Horizon amount (integer)", min_value=1, max_value=100000, value=10)
#                 if st.button("Run analysis"):
#                     if freq == 'M':
#                         big['period'] = big['Date'].dt.to_period('M').dt.to_timestamp()
#                     elif freq == 'Y':
#                         big['period'] = big['Date'].dt.to_period('A').dt.to_timestamp()
#                     else:
#                         big['period'] = big['Date']

#                     agg = big.groupby('period')[['Oil (BOPD)', 'Gas (MMSCFD)']].sum().reset_index().rename(columns={'period':'Date'})
#                     agg['Date'] = pd.to_datetime(agg['Date'])
#                     st.session_state['agg_cache'] = agg.copy()  # cache for KNN section
#                     st.subheader('Aggregated series')

#                     # Apply deferments to aggregated series for visualization
#                     deferments = st.session_state.get("deferments", {})
#                     agg_adj = apply_deferments(agg, deferments)

#                     # Apply scheduled ramp-ups/declines to aggregated (historical) series for visualization
#                     # Use st.session_state ramp/decline dicts directly; function will interpret the dict structure
#                     agg_adj = apply_scheduled_changes(agg_adj, st.session_state.get("rampups", {}), "Oil (BOPD)")
#                     agg_adj = apply_scheduled_changes(agg_adj, st.session_state.get("declines", {}), "Oil (BOPD)")
#                     agg_adj = apply_scheduled_changes(agg_adj, st.session_state.get("rampups", {}), "Gas (MMSCFD)")
#                     agg_adj = apply_scheduled_changes(agg_adj, st.session_state.get("declines", {}), "Gas (MMSCFD)")

#                     # The target average stored (if any) should be applied optionally for display
#                     ta = st.session_state.get("target_avg", {})
#                     if ta:
#                         if ta.get("Oil"):
#                             agg_adj = apply_target_average(agg_adj, target_avg=ta.get("Oil"), column="Oil (BOPD)", lock_schedules=ta.get("lock", True))
#                         if ta.get("Gas"):
#                             agg_adj = apply_target_average(agg_adj, target_avg=ta.get("Gas"), column="Gas (MMSCFD)", lock_schedules=ta.get("lock", True))

#                     st.line_chart(agg_adj.set_index('Date')[['Oil (BOPD)', 'Gas (MMSCFD)']])

#                     # Matplotlib plot with shading (historical)
#                     fig, ax = plt.subplots(figsize=(10,4))
#                     ax.plot(agg_adj['Date'], agg_adj['Oil (BOPD)'], label='Oil (BOPD)')
#                     ax.plot(agg_adj['Date'], agg_adj['Gas (MMSCFD)'], label='Gas (MMSCFD)')
#                     shade_deferment_spans(ax, deferments)
#                     ax.set_xlabel('Date'); ax.set_ylabel('Production')
#                     ax.legend(); ax.grid(True)
#                     st.pyplot(fig)

#     # --- KNN extended forecasting (re-usable) ---
#     st.markdown("---")
#     st.subheader("KNN-based Extended Forecast")
#     knn_source_df = st.session_state.get("agg_cache", None)
#     if knn_source_df is None and not files_df.empty:
#         pick = st.selectbox("Pick single uploaded file for KNN (if no agg cached)", files_df['filename'].tolist(), key='knn_pick')
#         if pick:
#             fp = files_df.loc[files_df['filename']==pick,'filepath'].values[0]
#             if os.path.exists(fp):
#                 tmp = pd.read_csv(fp, parse_dates=['Date'])
#                 tmp = tmp.groupby('Date')[['Oil (BOPD)','Gas (MMSCFD)']].sum().reset_index()
#                 knn_source_df = tmp.sort_values('Date')
#     if knn_source_df is None or knn_source_df.empty:
#         st.info("No data available for KNN. Create/upload and aggregate first.")
#     else:
#         series_choice = st.radio("Model series", ["Oil (BOPD)", "Gas (MMSCFD)"], horizontal=True)
#         df_prod = knn_source_df[['Date', series_choice]].rename(columns={series_choice:'Production'})

#         # Apply deferments to the series
#         df_prod = apply_deferments(df_prod, st.session_state.get("deferments", {}))

#         if df_prod.empty:
#             st.warning("No data points to model.")
#         else:
#             if not SKLEARN_AVAILABLE:
#                 st.error("scikit-learn not available. `pip install scikit-learn` to enable KNN forecasting.")
#             else:
#                 with st.expander("KNN Settings"):
#                     max_val = int(df_prod['Production'].max() * 2) if df_prod['Production'].max() > 0 else 10000
#                     target_avg = st.slider("Target Average (BOPD/MMscfd)", 0, max_val, min(4500, max_val//2), 100)
#                     n_neighbors = st.slider("KNN neighbors", 1, 20, 3)
#                     extend_years = st.slider("Forecast horizon (years)", 1, 75, 10)

#                 df_prod = df_prod.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
#                 lock_end = df_prod['Date'].max()
#                 hist = df_prod.copy()
#                 hist['Days'] = (hist['Date'] - hist['Date'].min()).dt.days
#                 X = hist[['Days']]
#                 y = hist['Production'].fillna(0)
#                 knn = KNeighborsRegressor(n_neighbors=n_neighbors)
#                 knn.fit(X, y)

#                 future_end = lock_end + pd.DateOffset(years=int(extend_years))
#                 future_days = pd.date_range(start=lock_end + pd.Timedelta(days=1), end=future_end, freq='D')
#                 future_X = (future_days - hist['Date'].min()).days.values.reshape(-1,1)
#                 future_pred = knn.predict(future_X)
#                 future_df = pd.DataFrame({'Date': future_days, 'Production': future_pred})
#                 forecast_df = pd.concat([hist[['Date','Production']], future_df], ignore_index=True).reset_index(drop=True)

#                 # *** APPLY SCHEDULES TO KNN FORECAST & HISTORICAL PRODUCTION ***
#                 # Our schedules in session_state are dicts like {'ramp_1': {'date': 'YYYY-MM-DD', 'rate': 5000}, ...}
#                 # apply_scheduled_changes can accept these dicts and will compute absolute targets or convert pct declines.
#                 # Apply to historical part first (for display / editing)
#                 hist_part = forecast_df[forecast_df['Date'] <= lock_end].reset_index(drop=True)
#                 hist_part = apply_scheduled_changes(hist_part, st.session_state.get("rampups", {}), "Production")
#                 hist_part = apply_scheduled_changes(hist_part, st.session_state.get("declines", {}), "Production")

#                 # Then apply to forecast part
#                 fcst_part = forecast_df[forecast_df['Date'] > lock_end].reset_index(drop=True)
#                 # When schedules reference dates beyond history, apply_scheduled_changes will create/insert rows and interpolate
#                 fcst_part = apply_scheduled_changes(fcst_part, st.session_state.get("rampups", {}), "Production")
#                 fcst_part = apply_scheduled_changes(fcst_part, st.session_state.get("declines", {}), "Production")

#                 # Recombine and ensure ordering
#                 forecast_df = pd.concat([hist_part, fcst_part], ignore_index=True).sort_values('Date').reset_index(drop=True)

#                 # Apply deferments to forecast (set to 0)
#                 deferments = st.session_state.get("deferments", {})
#                 defer_dates = []
#                 for d in deferments.values():
#                     if d.get('reason') and d.get('reason') != 'None' and int(d.get('duration_days',0))>0:
#                         sd = pd.to_datetime(d.get('start_date'), errors='coerce')
#                         if pd.isna(sd):
#                             continue
#                         days = int(d.get('duration_days',0))
#                         defer_dates += pd.date_range(sd, periods=days, freq='D').tolist()
#                 if defer_dates:
#                     forecast_df['Deferred'] = forecast_df['Date'].isin(defer_dates)
#                     forecast_df.loc[forecast_df['Deferred'], 'Production'] = 0
#                 else:
#                     forecast_df['Deferred'] = False

#                 # Rescale future non-deferred days to hit target_avg (over whole forecast_df)
#                 total_days = len(forecast_df)
#                 if total_days>0:
#                     hist_mask = forecast_df['Date'] <= lock_end
#                     hist_cum = forecast_df.loc[hist_mask, 'Production'].sum()
#                     required_total = target_avg * total_days
#                     required_future_prod = required_total - hist_cum
#                     valid_future_mask = (forecast_df['Date'] > lock_end) & (~forecast_df['Deferred'])
#                     num_valid = valid_future_mask.sum()
#                     if num_valid > 0:
#                         new_avg = required_future_prod / num_valid
#                         forecast_df.loc[valid_future_mask, 'Production'] = new_avg

#                 # Now apply global target average if set in sidebar session (this keeps original behavior)
#                 ta = st.session_state.get("target_avg", {})
#                 if ta and ta.get("Oil") and series_choice == "Oil (BOPD)":
#                     forecast_df = apply_target_average(forecast_df, target_avg=ta.get("Oil"), column="Production", lock_schedules=ta.get("lock", True))
#                 if ta and ta.get("Gas") and series_choice == "Gas (MMSCFD)":
#                     forecast_df = apply_target_average(forecast_df, target_avg=ta.get("Gas"), column="Production", lock_schedules=ta.get("lock", True))

#                 forecast_df['Year'] = forecast_df['Date'].dt.year
#                 min_year = int(forecast_df['Year'].min())
#                 max_year = int(forecast_df['Year'].max())
#                 sel_years = st.slider("Select Year Range", min_year, max_year, (min_year, max_year))
#                 analysis_df = forecast_df[(forecast_df['Year'] >= sel_years[0]) & (forecast_df['Year'] <= sel_years[1])]

#                 # metrics & plot
#                 st.metric("Cumulative Production", f"{analysis_df['Production'].sum():,.0f}")
#                 fig, ax = plt.subplots(figsize=(12,4))
#                 ax.plot(analysis_df['Date'], analysis_df['Production'], label='Production', color="green")
#                 ax.axhline(target_avg, linestyle='--', label='Target Avg', color="red")
#                 ax.axvline(lock_end, linestyle='--', label='End of History', color="black")
#                 shade_deferment_spans(ax, deferments)
#                 ax.set_title(f"KNN Forecast — {series_choice}")
#                 ax.set_xlabel("Date"); ax.set_ylabel(series_choice)
#                 ax.legend(); ax.grid(True)
#                 st.pyplot(fig)

#                 # editable tables
#                 st.subheader("Edit Historical")
#                 historical_data = analysis_df[analysis_df['Date'] <= lock_end][['Date','Production','Deferred']]
#                 hist_edit = st.data_editor(historical_data, num_rows='dynamic', key='knn_hist_editor')
#                 st.subheader("Edit Forecast")
#                 forecast_only = analysis_df[analysis_df['Date'] > lock_end][['Date','Production','Deferred']]
#                 forecast_only = forecast_only[~forecast_only['Date'].isin(hist_edit['Date'])]
#                 fcst_edit = st.data_editor(forecast_only, num_rows='dynamic', key='knn_fcst_editor')

#                 merged = pd.concat([hist_edit, fcst_edit], ignore_index=True).sort_values('Date')
#                 st.subheader("Forecast Data (editable)")
#                 st.dataframe(merged, hide_index=True)

#                 # downloads
#                 csv_data = merged.to_csv(index=False).encode('utf-8')
#                 excel_buf = BytesIO()
#                 with pd.ExcelWriter(excel_buf, engine='xlsxwriter') as writer:
#                     merged.to_excel(writer, sheet_name='Forecast', index=False)
#                 st.download_button("Download CSV", data=csv_data, file_name='forecast.csv')
#                 st.download_button("Download Excel", data=excel_buf.getvalue(), file_name='forecast.xlsx')

# # ------------------ ACCOUNT & ADMIN ------------------
# with tabs[2]:
#     st.header("Account & Admin")
#     st.subheader("Account info")
#     st.write(user)
#     st.markdown("---")
#     if user['is_admin']:
#         st.subheader("Admin: view all uploads")
#         cur.execute("SELECT u.id,u.username,a.filename,a.uploaded_at,a.field_name,a.filepath FROM users u JOIN uploads a ON u.id=a.user_id ORDER BY a.uploaded_at DESC LIMIT 500")
#         allrows = cur.fetchall()
#         if allrows:
#             adf = pd.DataFrame(allrows, columns=['user_id','username','filename','uploaded_at','field_name','filepath'])
#             st.dataframe(adf)
#             if st.button("Download combined CSV of all uploads"):
#                 frames = []
#                 for fp in adf['filepath']:
#                     if os.path.exists(fp):
#                         frames.append(pd.read_csv(fp))
#                 if frames:
#                     combined = pd.concat(frames, ignore_index=True)
#                     buf = BytesIO()
#                     combined.to_csv(buf, index=False)
#                     buf.seek(0)
#                     st.download_button("Download combined CSV (final)", data=buf, file_name='combined_all_users.csv')
#                 else:
#                     st.warning("No files found on disk")
#         else:
#             st.info("No uploads yet")
#     else:
#         st.info("You are not an admin. Admins can view/download uploads.")

# st.sidebar.markdown('---')
# st.sidebar.write(f"Local DB: {DB_PATH}")

# import os

# # Directory where files are stored
# DATA_DIR = os.path.join(os.getcwd(), "uploaded_files")

# # ------------------ RECENT UPLOADS TAB ------------------
# with tabs[3]:
#     st.header("🕒 Recent Uploads")

#     if user["is_admin"]:
#         cur.execute("""
#             SELECT u.filename, u.field_name, u.notes, u.uploaded_at, usr.username
#             FROM uploads u
#             JOIN users usr ON u.user_id = usr.id
#             ORDER BY u.uploaded_at DESC
#             LIMIT 10
#         """)
#         rows = cur.fetchall()

#         if rows:
#             st.subheader("All Users' Recent Uploads (Admin)")
#             for idx, r in enumerate(rows):
#                 col1, col2, col3, col4 = st.columns([2,3,2,2])
#                 col1.write(r[0])   # filename
#                 col2.write(r[4])   # uploaded_by
#                 col3.write(r[3])   # uploaded_at

#                 # Use separate keys
#                 confirm_state_key = f"admin_confirm_state_{idx}"
#                 delete_button_key = f"admin_delete_btn_{idx}"
#                 confirm_button_key = f"admin_confirm_btn_{idx}"

#                 if st.session_state.get(confirm_state_key, False):
#                     if col4.button("❌ Confirm Delete", key=confirm_button_key):
#                         # Delete from DB
#                         cur.execute("DELETE FROM uploads WHERE filename=?", (r[0],))
#                         conn.commit()

#                         # Delete from disk
#                         file_path = os.path.join(DATA_DIR, r[0])
#                         if os.path.exists(file_path):
#                             os.remove(file_path)

#                         st.warning(f"Admin deleted {r[0]}")
#                         st.session_state[confirm_state_key] = False
#                         st.experimental_rerun()
#                 else:
#                     if col4.button("🗑️ Delete", key=delete_button_key):
#                         st.session_state[confirm_state_key] = True
#                         st.error(f"⚠️ Click confirm to delete {r[0]}")
#         else:
#             st.info("No recent uploads in the system.")

#     else:
#         cur.execute(
#             "SELECT filename, field_name, notes, uploaded_at FROM uploads WHERE user_id = ? ORDER BY uploaded_at DESC LIMIT 5",
#             (user['id'],)
#         )
#         rows = cur.fetchall()

#         if rows:
#             st.subheader("My Recent Uploads")
#             for idx, r in enumerate(rows):
#                 col1, col2, col3 = st.columns([3,3,2])
#                 col1.write(r[0])   # filename
#                 col2.write(r[3])   # uploaded_at

#                 # Use separate keys
#                 confirm_state_key = f"user_confirm_state_{idx}"
#                 delete_button_key = f"user_delete_btn_{idx}"
#                 confirm_button_key = f"user_confirm_btn_{idx}"

#                 if st.session_state.get(confirm_state_key, False):
#                     if col3.button("❌ Confirm Delete", key=confirm_button_key):
#                         # Delete from DB
#                         cur.execute("DELETE FROM uploads WHERE filename=? AND user_id=?", (r[0], user['id']))
#                         conn.commit()

#                         # Delete from disk
#                         file_path = os.path.join(DATA_DIR, r[0])
#                         if os.path.exists(file_path):
#                             os.remove(file_path)

#                         st.success(f"Deleted {r[0]}")
#                         st.session_state[confirm_state_key] = False
#                         st.experimental_rerun()
#                 else:
#                     if col3.button("🗑️ Delete", key=delete_button_key):
#                         st.session_state[confirm_state_key] = True
#                         st.error(f"⚠️ Click confirm to delete {r[0]}")
#         else:
#             st.info("You have no recent uploads.")





# # ------------------ SAVED FILES TAB ------------------
# with tabs[4]:  # adjust index depending on your layout query language
#     st.header("📂 Saved Files")

#     if user["is_admin"]:
#         cur.execute("""
#             SELECT u.filepath, u.filename, u.field_name, u.notes, u.uploaded_at, usr.username
#             FROM uploads u
#             JOIN users usr ON u.user_id = usr.id
#             ORDER BY u.uploaded_at DESC
#         """)
#         rows = cur.fetchall()

#         if rows:
#             df_files = pd.DataFrame(rows, columns=["Filepath", "Filename", "Field", "Notes", "Uploaded At", "Uploaded By"])
#             for idx, row in df_files.iterrows():
#                 st.markdown(f"**{row['Filename']}** — Field: {row['Field']} | By: {row['Uploaded By']} | {row['Uploaded At']}")
#                 try:
#                     with open(row["Filepath"], "rb") as f:
#                         st.download_button(
#                             label="⬇️ Download",
#                             data=f,
#                             file_name=row["Filename"],
#                             mime="text/csv",
#                             key=f"download_{idx}"
#                         )
#                 except FileNotFoundError:
#                     st.error(f"File {row['Filename']} not found on disk.")
#         else:
#             st.info("No files saved in the system yet.")
#     else:
#         cur.execute(
#             "SELECT filepath, filename, field_name, notes, uploaded_at FROM uploads WHERE user_id = ? ORDER BY uploaded_at DESC",
#             (user['id'],)
#         )
#         rows = cur.fetchall()

#         if rows:
#             df_files = pd.DataFrame(rows, columns=["Filepath", "Filename", "Field", "Notes", "Uploaded At"])
#             for idx, row in df_files.iterrows():
#                 st.markdown(f"**{row['Filename']}** — Field: {row['Field']} | {row['Uploaded At']}")
#                 try:
#                     with open(row["Filepath"], "rb") as f:
#                         st.download_button(
#                             label="⬇️ Download",
#                             data=f,
#                             file_name=row["Filename"],
#                             mime="text/csv",
#                             key=f"user_download_{idx}"
#                         )
#                 except FileNotFoundError:
#                     st.error(f"File {row['Filename']} not found on disk.")
#         else:
#             st.info("You have not saved any files yet.")
###################################################################################

# #####################################################################
# # production_manager_full.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import sqlite3
# import os
# import hashlib
# import time
# from datetime import datetime, timedelta
# from io import BytesIO
# import matplotlib.pyplot as plt


# # optional: KNN
# try:
#     from sklearn.neighbors import KNeighborsRegressor
#     SKLEARN_AVAILABLE = True
# except Exception:
#     SKLEARN_AVAILABLE = False

# # ------------------ CONFIG ------------------
# DB_PATH = "production_app.db"
# DATA_DIR = "data_uploads"
# os.makedirs(DATA_DIR, exist_ok=True)

# # ------------------ DB ------------------
# conn = sqlite3.connect(DB_PATH, check_same_thread=False)
# cur = conn.cursor()


# # # ------------------ DB ------------------
# # conn = sqlite3.connect(DB_PATH, check_same_thread=False)
# # cur = conn.cursor()


# # import sqlite3

# # DB_PATH = "production_app.db"

# # def reset_users_table():
# #     # Use a separate connection just for this operation
# #     with sqlite3.connect(DB_PATH) as conn:
# #         cur = conn.cursor()

# #         # Drop the table if it exists
# #         cur.execute("DROP TABLE IF EXISTS users")

# #         # Recreate the table
# #         cur.execute('''
# #         CREATE TABLE users (
# #             id INTEGER PRIMARY KEY AUTOINCREMENT,
# #             username TEXT UNIQUE,
# #             email TEXT UNIQUE,
# #             password_hash TEXT,
# #             is_admin INTEGER DEFAULT 0,
# #             is_verified INTEGER DEFAULT 0,
# #             verify_token TEXT,
# #             reset_token TEXT,
# #             created_at TEXT
# #         )
# #         ''')

# #         conn.commit()
# #         print("✅ Users table has been reset successfully!")

# # # Call this function once when you want to reset
# # reset_users_table()

# # --- Create users table if it doesn't exist ---
# cur.execute('''
# CREATE TABLE IF NOT EXISTS users (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     username TEXT UNIQUE,
#     email TEXT UNIQUE,
#     password_hash TEXT,
#     is_admin INTEGER DEFAULT 0,
#     created_at TEXT
# )
# ''')

# # --- Create uploads table if it doesn't exist ---
# cur.execute('''
# CREATE TABLE IF NOT EXISTS uploads (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     user_id INTEGER,
#     filename TEXT,
#     filepath TEXT,
#     uploaded_at TEXT,
#     field_name TEXT,
#     notes TEXT,
#     source TEXT DEFAULT 'upload',
#     FOREIGN KEY(user_id) REFERENCES users(id)
# )
# ''')

# conn.commit()

# # --- Ensure the users table has the additional columns ---
# def ensure_user_columns():
#     cur.execute("PRAGMA table_info(users)")
#     existing = [row[1] for row in cur.fetchall()]

#     if "is_verified" not in existing:
#         cur.execute("ALTER TABLE users ADD COLUMN is_verified INTEGER DEFAULT 0")
#     if "verify_token" not in existing:
#         cur.execute("ALTER TABLE users ADD COLUMN verify_token TEXT")
#     if "reset_token" not in existing:
#         cur.execute("ALTER TABLE users ADD COLUMN reset_token TEXT")
#     conn.commit()

# ensure_user_columns()

# # ------------------ HELPERS & NEW SCHEDULING LOGIC ------------------
# REQUIRED_COLS = ["Date", "Oil (BOPD)", "Gas (MMSCFD)"]

# def hash_password(password: str) -> str:
#     salt = "static_salt_change_me"  # replace in production
#     return hashlib.sha256((salt + password).encode()).hexdigest()

# def create_user(username, email, password, is_admin=0):
#     ph = hash_password(password)
#     try:
#         cur.execute("INSERT INTO users (username,email,password_hash,is_admin,created_at) VALUES (?,?,?,?,?)",
#                     (username, email, ph, int(is_admin), datetime.utcnow().isoformat()))
#         conn.commit()
#         return True, "user created"
#     except sqlite3.IntegrityError as e:
#         return False, str(e)


# def authenticate(username_or_email, password):
#     ph = hash_password(password)
#     cur.execute(
#         "SELECT id, username, email, is_admin, is_verified FROM users WHERE (username=? OR email=?) AND password_hash=?",
#         (username_or_email, username_or_email, ph)
#     )
#     row = cur.fetchone()


#     cur.execute("SELECT id, username, email, is_verified FROM users")
#     print(cur.fetchall())

#     if row:
#         user = {
#             "id": row[0],
#             "username": row[1],
#             "email": row[2],
#             "is_admin": row[3],
#             "is_verified": row[4]  # ✅ include this!
#         }
#         return True, user
#     else:
#         return False, None

# def validate_and_normalize_df(df: pd.DataFrame):
#     # Normalize column names and check required
#     colmap = {}
#     for c in df.columns:
#         c_clean = c.strip()
#         if c_clean.lower() in [x.lower() for x in REQUIRED_COLS]:
#             for req in REQUIRED_COLS:
#                 if c_clean.lower() == req.lower():
#                     colmap[c] = req
#         elif 'date' in c_clean.lower():
#             colmap[c] = 'Date'
#         elif 'oil' in c_clean.lower():
#             colmap[c] = 'Oil (BOPD)'
#         elif 'gas' in c_clean.lower():
#             colmap[c] = 'Gas (MMSCFD)'
#         else:
#             colmap[c] = c_clean
#     df = df.rename(columns=colmap)
#     missing = [c for c in REQUIRED_COLS if c not in df.columns]
#     if missing:
#         return False, f"Missing required columns: {missing}"
#     try:
#         df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
#     except Exception as e:
#         return False, f"Date parse error: {e}"
#     if df['Date'].isna().any():
#         return False, "Some Date values could not be parsed. Ensure ISO-like dates (YYYY-MM-DD)."
#     for c in ['Oil (BOPD)','Gas (MMSCFD)']:
#         df[c] = pd.to_numeric(df[c], errors='coerce')
#         if df[c].isna().all():
#             return False, f"Column {c} has no numeric values"
#     return True, df

# def save_dataframe(user_id, df: pd.DataFrame, field_name: str, notes: str = "", source: str = "upload"):
#     ts = int(time.time())
#     safe_field = field_name.replace(" ", "_")[:60] if field_name else "unnamed"
#     filename = f"user{user_id}_{safe_field}_{ts}.csv"
#     filepath = os.path.join(DATA_DIR, filename)
#     df.to_csv(filepath, index=False)
#     cur.execute("INSERT INTO uploads (user_id,filename,filepath,uploaded_at,field_name,notes,source) VALUES (?,?,?,?,?,?,?)",
#                 (user_id, filename, filepath, datetime.utcnow().isoformat(), field_name, notes, source))
#     conn.commit()
#     return filename, filepath

# def apply_deferments(df, deferments):
#     """Apply deferments by zeroing Oil & Gas in deferment ranges."""
#     if df is None or df.empty or not deferments:
#         return df
#     df = df.copy()
#     if 'Date' not in df.columns:
#         return df
#     df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
#     for d in deferments.values():
#         reason = d.get('reason')
#         days = int(d.get('duration_days', 0) or 0)
#         start_str = d.get('start_date')
#         if reason and reason != "None" and days > 0 and start_str:
#             try:
#                 start = pd.to_datetime(start_str, errors='coerce')
#                 if pd.isna(start):
#                     continue
#                 end = start + pd.Timedelta(days=days-1)
#                 mask = (df['Date'] >= start) & (df['Date'] <= end)
#                 if 'Oil (BOPD)' in df.columns:
#                     df.loc[mask, 'Oil (BOPD)'] = 0
#                 if 'Gas (MMSCFD)' in df.columns:
#                     df.loc[mask, 'Gas (MMSCFD)'] = 0
#                 if 'Production' in df.columns:
#                     df.loc[mask, 'Production'] = 0
#             except Exception:
#                 continue
#     return df

# def shade_deferment_spans(ax, deferments, color_map=None):
#     spans = []
#     for d in deferments.values():
#         reason = d.get('reason')
#         days = int(d.get('duration_days', 0) or 0)
#         start_str = d.get('start_date')
#         if reason and reason != "None" and days > 0 and start_str:
#             s = pd.to_datetime(start_str, errors='coerce')
#             if pd.isna(s):
#                 continue
#             e = s + pd.Timedelta(days=days-1)
#             spans.append((s, e, reason))
#     for s, e, r in spans:
#         color = None
#         if color_map and r in color_map:
#             color = color_map[r]
#         ax.axvspan(s, e, alpha=0.18, color=color)

# # ------------------ NEW: SCHEDULING & TARGET AVERAGE HELPERS ------------------

# def _ensure_date_index(df):
#     """Return a copy indexed by Date (datetime), preserving original Date column too."""
#     d = df.copy()
#     if 'Date' in d.columns:
#         d['Date'] = pd.to_datetime(d['Date'], errors='coerce')
#         d = d.sort_values('Date').reset_index(drop=True)
#         d_indexed = d.set_index('Date', drop=False)
#     else:
#         # Try to use existing index
#         idx = pd.to_datetime(d.index)
#         d_indexed = d.copy()
#         d_indexed['Date'] = idx
#         d_indexed.index = idx
#     return d_indexed

# def apply_scheduled_changes(df, schedules, column):
#     """
#     Apply scheduled ramp-ups/declines with linear interpolation on the specified column.
#     schedules: dict where values are dicts (as stored in session_state) OR a mapping date->target
#     Accepts input df that has a 'Date' column or datetime index. Returns df with same shape.
#     """
#     if df is None or df.empty or not schedules or column not in df.columns:
#         return df

#     # Convert to DataFrame indexed by Date
#     df_i = _ensure_date_index(df)

#     # Accept two possible schedule formats:
#     # 1) state dict as in session_state: {'ramp_1': {'date':'YYYY-MM-DD','rate':5000}, ...}
#     # 2) direct mapping: {'2025-09-01': 5000, ...}
#     mapping = {}
#     # if schedules is a mapping of date->value (string/number)
#     if all((isinstance(k, (str, pd.Timestamp)) and not isinstance(v, dict)) for k, v in schedules.items()):
#         # treat as direct mapping
#         for k, v in schedules.items():
#             try:
#                 d = pd.to_datetime(k)
#                 mapping[d] = float(v)
#             except Exception:
#                 continue
#     else:
#         # treat as state dict
#         for entry in schedules.values():
#             try:
#                 d = pd.to_datetime(entry.get('date'))
#             except Exception:
#                 continue
#             if pd.isna(d):
#                 continue
#             # If entry has 'rate' it's absolute target
#             if 'rate' in entry:
#                 try:
#                     mapping[d] = float(entry.get('rate'))
#                 except Exception:
#                     continue
#             # If entry has 'pct' treat as a percent change to the base value at that date
#             elif 'pct' in entry:
#                 try:
#                     pct = float(entry.get('pct')) / 100.0
#                 except Exception:
#                     pct = 0.0
#                 # find base value just before or at that date
#                 if d in df_i.index:
#                     base = df_i.loc[d, column]
#                     if pd.isna(base):
#                         # take previous non-na
#                         prev = df_i.loc[:d, column].ffill().iloc[-1] if not df_i.loc[:d, column].ffill().empty else None
#                         base = prev if prev is not None else 0.0
#                 else:
#                     # take last known before date
#                     prev_series = df_i.loc[df_i.index <= d, column]
#                     if not prev_series.empty:
#                         base = prev_series.ffill().iloc[-1]
#                     else:
#                         # fallback: use earliest value
#                         base = df_i[column].ffill().iloc[0] if not df_i[column].ffill().empty else 0.0
#                 mapping[d] = float(base * (1 - pct))

#     if not mapping:
#         return df

#     # Keep only mapping dates within the df index range (we can still insert future dates if df extends into forecast)
#     # Insert scheduled targets (create rows if necessary)
#     for d, target in mapping.items():
#         if d not in df_i.index:
#             # if date outside range, add a new row so interpolation can happen
#             # create a new row using NaN for all columns, set Date index to d
#             new_row = pd.DataFrame([{c: np.nan for c in df_i.columns}])
#             new_row.index = pd.Index([d])
#             new_row['Date'] = d
#             df_i = pd.concat([df_i, new_row])
#     # Re-sort index
#     df_i = df_i.sort_index()

#     # Set scheduled absolute values
#     for d, target in mapping.items():
#         df_i.at[d, column] = target

#     # Interpolate the column across the entire index (linear)
#     df_i[column] = df_i[column].interpolate(method='time')  # time-aware interpolation

#     # Return df_i with original order and with Date column preserved
#     # Remove any rows we artificially added that were outside original df's full date coverage only if original had no such dates
#     # To be safe, convert back to same index shape as original input by reindexing to original index if original had explicit index
#     out = df_i.reset_index(drop=True)
#     # Recreate the original shape: ensure it has same number of rows as before if original index used; but we often want the scheduled rows kept
#     # We'll return df_i.reset_index(drop=True) but keep Date column (so other code that groups by Date still works)
#     return df_i.reset_index(drop=True).copy()

# def apply_target_average(df, target_avg=None, column=None, lock_schedules=False):
#     """
#     Scale a column in df to match target_avg.
#     If lock_schedules=True, it will try to preserve points where values were explicitly set (i.e., large jumps).
#     Note: target_avg expected to be average over the dataframe (not per-day).
#     """
#     if df is None or df.empty or column not in df.columns or target_avg is None:
#         return df
#     d = df.copy()
#     cur_avg = d[column].mean()
#     if cur_avg <= 0:
#         return d
#     scale = float(target_avg) / float(cur_avg)
#     if not lock_schedules:
#         d[column] = d[column] * scale
#     else:
#         # Attempt to detect scheduled points as places with sudden changes (non-small diffs)
#         diffs = d[column].diff().abs()
#         # threshold: use median*5 or minimal fallback
#         thresh = max(diffs.median() * 5.0, 1e-6)
#         scheduled_idx = diffs[diffs > thresh].index.tolist()
#         # Scale only indices that are not scheduled
#         mask = ~d.index.isin(scheduled_idx)
#         d.loc[mask, column] = d.loc[mask, column] * scale
#     return d

# # ------------------ UI SETUP ------------------
# st.set_page_config(page_title="Production Manager", layout="wide")
# if 'auth' not in st.session_state:
#     st.session_state['auth'] = {'logged_in': False}

# #########################################  LOGIN SECTION  #####################################
# import smtplib
# import hashlib, time, os
# from email.mime.text import MIMEText
# import hashlib, time, os
# from datetime import datetime, timedelta
# from email.mime.multipart import MIMEMultipart
# import streamlit as st





# def send_email(to_email, subject, body):
#     sender_email = "hafsatuxy@gmail.com"   # must be verified in Brevo
#     password = "OKnmRy6V7fUY509I"
#     smtp_server = "smtp-relay.brevo.com"
#     smtp_port = 587

#     try:
#         msg = MIMEMultipart()
#         msg["From"] = sender_email
#         msg["To"] = to_email
#         msg["Subject"] = subject
#         msg.attach(MIMEText(body, "plain"))

#         server = smtplib.SMTP(smtp_server, smtp_port)
#         server.starttls()
#         server.login("96cf99001@smtp-brevo.com", password)  # login always with Brevo login
#         server.sendmail(sender_email, to_email, msg.as_string())
#         server.quit()

#         print("✅ Email sent successfully!")
#         return True
#     except Exception as e:
#         print("❌ Failed to send email:", e)
#         return False

# # ------------------ SIDEBAR: AUTH, DURATION ------------------
# st.sidebar.header("Account")

# # action = st.sidebar.selectbox("Action", ["Login", "Sign up", "Forgot password", "Verify account"])
# action = st.sidebar.selectbox(
#     "Action",
#     ["Login", "Sign up", "Forgot password", "Verify account", "Delete account"]
# )


# # ---------- SIGN UP ----------
# # Ensure user is always defined to avoid NameError
# user = st.session_state.get('auth', {}).get('user', None)





# def create_user(username, email, password, is_admin=0, is_verified=0):
#     password_hash = hash_password(password)
#     try:
#         cur.execute(
#             """
#             INSERT INTO users (username, email, password_hash, is_admin, is_verified, verify_token, reset_token)
#             VALUES (?, ?, ?, ?, ?, NULL, NULL)
#             """,
#             (username, email, password_hash, is_admin, is_verified)
#         )
#         conn.commit()
#         return True, "User created successfully"
#     except Exception as e:
#         return False, str(e)







# if action == 'Sign up':
#     st.sidebar.markdown("Create a new account")
#     su_user = st.sidebar.text_input("Username", key='su_user')
#     su_email = st.sidebar.text_input("Email", key='su_email')
#     su_pass = st.sidebar.text_input("Password", type='password', key='su_pass')
#     su_pass2 = st.sidebar.text_input("Confirm password", type='password', key='su_pass2')
    
#     if st.sidebar.button("Create account", key="create_account_btn"):
#         if su_pass != su_pass2:
#             st.sidebar.error("Passwords do not match")
#         elif not su_user or not su_pass or not su_email:
#             st.sidebar.error("Fill all fields")
#         else:
#             # 🚫 Force all signups as normal users
#             is_admin_flag = 0  

#             ok, msg = create_user(
#                 su_user, su_email, su_pass,
#                 is_admin=is_admin_flag,
#                 is_verified=0
#             )

#             if ok:
#                 # Generate verification token
#                 token = hashlib.sha1((su_email + str(time.time())).encode()).hexdigest()[:8]
#                 cur.execute("UPDATE users SET verify_token=? WHERE email=?", (token, su_email))
#                 conn.commit()

#                 # Send email with token
#                 send_email(
#                     su_email,
#                     "Verify Your Account - Production App",
#                     f"Hello {su_user},\n\nThank you for signing up.\n\n"
#                     f"Your verification code is: {token}\n\n"
#                     f"Enter this code in the app to activate your account."
#                 )
#                 st.success(f"✅ Signup successful! A verification email was sent to {su_email}. Please check your inbox (or spam).")
#             else:
#                 st.sidebar.error(msg)

# # ---------- VERIFY ACCOUNT ----------
# elif action == "Verify account":
#     st.sidebar.markdown("Enter the code sent to your email")
#     ver_email = st.sidebar.text_input("Email", key="ver_email")
#     ver_code = st.sidebar.text_input("Verification code", key="ver_code")

#     if st.sidebar.button("Verify now", key="verify_btn"):
#         cur.execute("SELECT id, verify_token FROM users WHERE email=?", (ver_email,))
#         r = cur.fetchone()
#         if r and r[1] == ver_code:
#             cur.execute("UPDATE users SET is_verified=1, verify_token=NULL WHERE id=?", (r[0],))
#             conn.commit()
#             st.sidebar.success("✅ Account verified. You can now login.")
#         else:
#             st.sidebar.error("❌ Invalid verification code")

#     # 🔹 Send verification code
#     if st.sidebar.button("Send Verification Code", key="resend_btn"):
#         cur.execute("SELECT verify_token FROM users WHERE email=?", (ver_email,))
#         r = cur.fetchone()

#         if r:
#             code = r[0]
#             subject = "Your Verification Code"
#             body = f"Hello,\n\nYour verification code is: {code}\n\nUse this to verify your account."
#             try:
#                 send_email(ver_email, subject, body)
#                 st.sidebar.success("📩 Verification code sent to your email!")
#             except Exception as e:
#                 st.sidebar.error(f"❌ Failed to send code: {e}")
#         else:
#             st.sidebar.warning("⚠️ Email not found. Please sign up first.")

# # ---------- FORGOT PASSWORD ----------
# elif action == 'Forgot password':
#     st.sidebar.markdown("Reset password")
#     fp_email = st.sidebar.text_input("Enter your email", key='fp_email')

#     if st.sidebar.button("Send reset code", key="reset_code_btn"):
#         cur.execute("SELECT id FROM users WHERE email=?", (fp_email,))
#         r = cur.fetchone()
#         if r:
#             token = hashlib.sha1((fp_email + str(time.time())).encode()).hexdigest()[:8]

#             cur.execute("UPDATE users SET reset_token=? WHERE id=?", (token, r[0]))
#             conn.commit()

#             send_email(
#                 fp_email,
#                 "Password Reset - Production App",
#                 f"Your reset code is: {token}\n\n"
#                 "Use this code in the app to set a new password."
#             )
#             st.sidebar.success("A reset code has been sent to your email.")
#         else:
#             st.sidebar.error("Email not found")

#     st.sidebar.markdown("---")
#     token_user = st.sidebar.text_input("User ID (from your account)", key="token_user")
#     token_val = st.sidebar.text_input("Reset code", key="token_val")
#     token_newpw = st.sidebar.text_input("New password", type='password', key="token_newpw")

#     if st.sidebar.button("Reset password now", key="reset_now_btn"):
#         try:
#             uid = int(token_user)
#             cur.execute("SELECT reset_token FROM users WHERE id=?", (uid,))
#             db_token = cur.fetchone()
#             if db_token and db_token[0] == token_val:
#                 ph = hash_password(token_newpw)
#                 cur.execute("UPDATE users SET password_hash=?, reset_token=NULL WHERE id=?", (ph, uid))
#                 conn.commit()
#                 st.sidebar.success("Password updated. Please login.")
#             else:
#                 st.sidebar.error("Invalid or expired token")
#         except Exception:
#             st.sidebar.error("Invalid user id")

# # ---------- LOGIN ----------
# elif action == "Login":
#     login_user = st.sidebar.text_input("Username or email", key='login_user')
#     login_pass = st.sidebar.text_input("Password", type='password', key='login_pass')

#     dur_map = {
#         "1 minute": 60, "5 minutes": 300, "30 minutes": 1800,
#         "1 hour": 3600, "1 day": 86400, "7 days": 604800, "30 days": 2592000
#     }
#     sel_dur = st.sidebar.selectbox("Login duration", list(dur_map.keys()), index=3)

#     if st.sidebar.button("Login", key="login_btn"):
#         ok, user = authenticate(login_user, login_pass)
#         if ok:
#             # ✅ Ensure verified check works properly
#             if not user.get('is_verified') or user['is_verified'] == 0:
#                 st.sidebar.error("⚠️ Please verify your email before logging in.")
#             else:
#                 st.session_state['auth'] = {
#                     'logged_in': True,
#                     'user': user,
#                     'expires_at': (datetime.utcnow() + timedelta(seconds=dur_map[sel_dur])).timestamp()
#                 }
#                 # st.sidebar.success(f"✅ Logged in as {user['username']}")

#                 # Send login notification email
#                 send_email(
#                     user['email'],
#                     "Login Notification - Production App",
#                     f"Hello {user['username']},\n\n"
#                     f"Your account was just logged in on {datetime.utcnow()} (UTC)."
#                 )
#         else:
#             st.sidebar.error("❌ Invalid credentials. If you are a new user, please sign up.")



#     # # --- After login/signup, check session ---
#     # auth = st.session_state.get("auth", {})
#     # user = auth.get("user")

#     # Add these two lines (safe global access to session_state auth)
#     auth = st.session_state.get("auth", {})
#     user = auth.get("user")

#     if not auth.get("logged_in"):
#         # Not logged in yet
#         st.title("Welcome — To the Production App")
#         st.write("Please sign up or login from the sidebar.")
#         st.stop()  # 🔴 Prevents the rest of the app from running
#     else:
#         # ✅ Logged in successfully → safe to use user['id']
#         st.sidebar.success(f"Logged in as {user['username']}")

#         # Example: show uploads for this user
#         cur.execute(
#             "SELECT id, filename, filepath, field_name, uploaded_at FROM uploads WHERE user_id=? ORDER BY uploaded_at DESC",
#             (user['id'],) if user else (-1,)

#         )
#         uploads = cur.fetchall()

#         st.write("### Your Uploads")
#         # if uploads:
#         #     for up in uploads:
#         #         st.write(f"📂 {up[1]} (uploaded {up[4]})")
#         # else:
#         #     st.write("No uploads yet.")
#         if uploads:
#            df = pd.DataFrame(uploads, columns=["ID", "Filename", "Path", "Field", "Uploaded At"])
#            st.dataframe(df)
#         else:
#             st.write("No uploads yet.")



#         if st.sidebar.button("Logout", key="logout_btn"):
#             st.session_state["auth"] = {"logged_in": False, "user": None}
#             st.experimental_rerun()

# elif action == "Delete account":
#     st.sidebar.markdown("Delete your account permanently")

#     del_user = st.sidebar.text_input("Username or Email", key="del_user")
#     del_pass = st.sidebar.text_input("Password", type="password", key="del_pass")

#     if st.sidebar.button("Delete my account", key="del_btn"):
#         ok, user = authenticate(del_user, del_pass)
#         if ok and user:
#             # Delete account from DB
#             cur.execute("DELETE FROM users WHERE id=?", (user['id'],) if user else (-1,)
# )
#             conn.commit()

#             # Clear session
#             if "auth" in st.session_state:
#                 st.session_state.pop("auth")

#             st.sidebar.success("Your account has been permanently deleted.")
#             st.session_state["user"] = None

#             # Optional: send email notification
#             send_email(
#                 user['email'],
#                 "Account Deleted - Production App",
#                 f"Hello {user['username']},\n\nYour account has been permanently deleted."
#             )

#         else:
#             st.sidebar.error("Invalid username/email or password")

# # ------------------ SIDEBAR: Deferments (100) ------------------
# with st.sidebar.expander("⚙️ Deferment Controls", expanded=False):
#     if "deferments" not in st.session_state:
#         st.session_state["deferments"] = {}

#     options = ["None", "Maintenance", "Pipeline Issue", "Reservoir", "Other"]
#     for i in range(1, 6):  # reduced to 5 in UI for practicality; original allowed 100. adjust as needed
#         st.markdown(f"**Deferment {i}**")
#         reason = st.selectbox(f"Reason {i}", options, key=f"deferment_reason_{i}")
#         start_date = st.date_input(f"Start Date {i}", key=f"deferment_start_{i}")
#         duration = st.number_input(f"Duration (days) {i}", min_value=0, max_value=3650, step=1, key=f"deferment_duration_{i}")
#         st.session_state["deferments"][f"Deferment {i}"] = {
#             "reason": reason,
#             "start_date": str(start_date),
#             "duration_days": int(duration)
#         }

#     st.markdown("### 📋 Active Deferments")
#     active = {k:v for k,v in st.session_state["deferments"].items() if v["reason"] != "None" and v["duration_days"] > 0}
#     if active:
#         for name, d in active.items():
#             st.markdown(f"- **{name}** → {d['reason']} starting {d['start_date']} for {d['duration_days']} days")
#     else:
#         st.info("No active deferments set")

# # ===========================
# # SIDEBAR CONTROLS - RAMP & DECLINE
# # ===========================
# with st.sidebar.expander("⚡ Ramp-up & Decline Controls", expanded=False):
#     if "rampups" not in st.session_state:
#         st.session_state["rampups"] = {}
#     if "declines" not in st.session_state:
#         st.session_state["declines"] = {}
#     # input for a single ramp or decline then add to session_state (keeps original UI)
#     ramp_date = st.date_input("Ramp-up Start Date", key="ramp_date")
#     ramp_rate = st.number_input("New Ramp-up Rate (absolute)", min_value=0.0, value=5000.0, key="ramp_rate")
#     if st.button("Apply Ramp-up"):
#         st.session_state["rampups"][f"ramp_{len(st.session_state['rampups'])+1}"] = {
#             "date": str(ramp_date),
#             "rate": float(ramp_rate)
#         }
#         st.success(f"Ramp-up set: {ramp_date} → {ramp_rate}")

#     st.markdown("---")
#     decline_date = st.date_input("Decline Start Date", key="decline_date")
#     decline_pct = st.slider("Decline Percentage (%)", 0, 100, 10, key="decline_pct")
#     if st.button("Apply Decline"):
#         st.session_state["declines"][f"decline_{len(st.session_state['declines'])+1}"] = {
#             "date": str(decline_date),
#             "pct": int(decline_pct)
#         }
#         st.success(f"Decline set: {decline_date} → -{decline_pct}%")

#     st.markdown("---")
#     st.write("Active Ramps")
#     for k, v in st.session_state.get("rampups", {}).items():
#         st.write(f"- {k}: {v.get('date')} → {v.get('rate')}")

#     st.write("Active Declines")
#     for k, v in st.session_state.get("declines", {}).items():
#         st.write(f"- {k}: {v.get('date')} → -{v.get('pct')}%")

#     st.markdown("---")
#     st.write
#     ("Target avg (optional) — applied after scheduling")
#     if "target_avg" not in st.session_state:
#         st.session_state["target_avg"] = {"Oil": None, "Gas": None, "lock": True}
#     ta_oil = st.number_input("Target Avg Oil (BOPD) — leave 0 to ignore", min_value=0, value=int(st.session_state["target_avg"].get("Oil") or 0), key="ta_oil")
#     ta_gas = st.number_input("Target Avg Gas (MMSCFD) — leave 0 to ignore", min_value=0, value=int(st.session_state["target_avg"].get("Gas") or 0), key="ta_gas")
#     lock_sched = st.checkbox("Lock scheduled points (do not rescale them)", value=st.session_state["target_avg"].get("lock", True))
#     if st.button("Set Target Avg"):
#         st.session_state["target_avg"]["Oil"] = int(ta_oil) if ta_oil > 0 else None
#         st.session_state["target_avg"]["Gas"] = int(ta_gas) if ta_gas > 0 else None
#         st.session_state["target_avg"]["lock"] = bool(lock_sched)
#         st.success("Target averages stored in session")




# # --- After ALL auth actions (login/signup/verify/forgot/delete), check login state ---
# auth = st.session_state.get("auth", {})
# user = auth.get("user")

# if not auth.get("logged_in"):
#     # Not logged in yet
#     st.title("Welcome — To the Production App")
#     st.write("Please sign up or login from the sidebar.")
#     st.stop()  # 🔴 Prevents the rest of the app from running
# else:
#     # Logged in successfully → safe to use user['id']
#     # st.sidebar.success(f"Logged in as {user['username']}")

#     # Example: show uploads for this user
#     cur.execute(
#         "SELECT id, filename, filepath, field_name, uploaded_at FROM uploads WHERE user_id=? ORDER BY uploaded_at DESC",
#         (user['id'],) if user else (-1,)

#     )
#     uploads = cur.fetchall()

#     st.write("### Your Uploads")
#     # if uploads:
#     #     for up in uploads:
#     #         st.write(f"📂 {up[1]} (uploaded {up[4]})")
#     # else:
#     #     st.write("No uploads yet.")
#     if uploads:
#         df = pd.DataFrame(uploads, columns=["ID", "Filename", "Path", "Field", "Uploaded At"])
#         st.dataframe(df)
#     else:
#         st.write("No uploads yet.")

# # --- Final global auth check ---
# auth = st.session_state.get("auth", {})
# user = auth.get("user")

# if not auth.get("logged_in"):
#     st.title("Welcome — To the Production App")
#     st.write("Please sign up or login from the sidebar.")
#     st.stop()
# else:
#     st.sidebar.success(f"✅ Logged in as {user['username']}")

#     # Admin-only DB path
#     if user and user.get("is_admin"):
#         st.sidebar.write(f"Local DB: {DB_PATH}")

#     # Show uploads
#     cur.execute(
#         "SELECT id, filename, filepath, field_name, uploaded_at FROM uploads WHERE user_id=? ORDER BY uploaded_at DESC",
#         (user['id'],)
#     )
#     uploads = cur.fetchall()

#     st.write("### Your Uploads")
#     if uploads:
#         df = pd.DataFrame(uploads, columns=["ID","Filename","Path","Field","Uploaded At"])
#         st.dataframe(df)
#     else:
#         st.write("No uploads yet.")

# ########################
# # If Admin, allow them to switch into a user's workspace
# if user and user.get("is_admin"):
#     st.sidebar.markdown("### 👥 Admin Workspace Switch")
#     cur.execute("SELECT id, username FROM users ORDER BY username ASC")
#     all_users = cur.fetchall()
#     user_map = {u[1]: u[0] for u in all_users}

#     selected_username = st.sidebar.selectbox(
#         "Select user workspace to view",
#         ["-- My own (Admin) --"] + list(user_map.keys())
#     )

#     if selected_username != "-- My own (Admin) --":
#         cur.execute("SELECT * FROM users WHERE id=?", (user_map[selected_username],))
#         impersonated_user = cur.fetchone()
#         if impersonated_user:
#             # Override `user` object
#             user = dict(zip([d[0] for d in cur.description], impersonated_user))

#             # 🔔 Banner across the app
#             st.markdown(
#                 f"""
#                 <div style="background-color:#ffecb3;
#                             padding:10px;
#                             border-radius:8px;
#                             border:1px solid #f0ad4e;
#                             margin-bottom:15px;">
#                     ⚠️ <b>Admin Mode:</b> You are impersonating <b>{user['username']}</b>.
#                     All actions (uploads, deletes, deferments) will be performed as this user.
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )


# # ------------------ MAIN TABS ------------------
# st.title("Production App — Workspaces")
# tabs = st.tabs(["Production Input", "Forecast Analysis", "Account & Admin", "Recent Uploads", "Saved Files"])

# # ------------------ PRODUCTION INPUT ------------------
# with tabs[0]:
#     st.header("Production Input")
#     st.write("Upload CSV files or do manual entry. Required columns: Date, Oil (BOPD), Gas (MMSCFD).")

#     # Let user choose method
#     method = st.radio("Choose Input Method", ["Upload CSV", "Manual Entry"], horizontal=True)

#     if method == "Upload CSV":
#         uploaded = st.file_uploader("Upload CSV", type=['csv'], key='upl1')
#         field_name = st.text_input("Field name (e.g. OML 98)")
#         notes = st.text_area("Notes (optional)")
#         if st.button("Upload and validate"):
#             if uploaded is None:
#                 st.error("Please select a file to upload.")
#             else:
#                 try:
#                     df = pd.read_csv(uploaded)
#                 except Exception as e:
#                     st.error(f"Could not read CSV: {e}")
#                     df = None
#                 if df is not None:
#                     ok, out = validate_and_normalize_df(df)
#                     if not ok:
#                         st.error(out)
#                     else:
#                         df_clean = out
#                         ts = int(time.time())
#                         filename = f"{user['username']}_{field_name}_{ts}.csv"
#                         filepath = os.path.join(DATA_DIR, filename)
#                         df_clean.to_csv(filepath, index=False)
#                         cur.execute("INSERT INTO uploads (user_id,filename,filepath,uploaded_at,field_name,notes) VALUES (?,?,?,?,?,?)",
#                                     (user['id'], filename, filepath, datetime.utcnow().isoformat(), field_name, notes))
#                         conn.commit()
#                         st.success("Uploaded and saved")
#                         st.dataframe(df_clean.head())

#     else:  # Manual Entry
#         st.subheader("Manual Excel-like Workspace (Paste or Type)")

#         ws_name = st.text_input("Workspace name", key="manual_ws_name")

#         if "manual_table" not in st.session_state:
#             st.session_state["manual_table"] = pd.DataFrame({
#                 "Date": [""] * 8,
#                 "Oil (BOPD)": [None] * 8,
#                 "Gas (MMSCFD)": [None] * 8
#             })

#         st.info("You can paste large blocks (Date, Oil, Gas). Date format YYYY-MM-DD recommended.")
#         edited_df = st.data_editor(st.session_state["manual_table"], num_rows="dynamic", key="manual_table_editor", use_container_width=True)
#         st.session_state["manual_table"] = edited_df

#         if not edited_df.empty:
#             st.subheader(f"Preview — {ws_name if ws_name else 'Unnamed'}")
#             st.dataframe(edited_df.head(50))

#             totals = pd.DataFrame({
#                 "Date": ["TOTAL"],
#                 "Oil (BOPD)": [pd.to_numeric(edited_df["Oil (BOPD)"], errors="coerce").sum(skipna=True)],
#                 "Gas (MMSCFD)": [pd.to_numeric(edited_df["Gas (MMSCFD)"], errors="coerce").sum(skipna=True)]
#             })
#             st.write("**Totals:**")
#             st.dataframe(totals)

#             if st.button("Save Workspace to CSV"):
#                 if not ws_name.strip():
#                     st.error("Please enter a workspace name.")
#                 else:
#                     ok, norm = validate_and_normalize_df(edited_df)
#                     if not ok:
#                         st.error(norm)
#                     else:
#                         ts = int(time.time())
#                         filename = f"{user['username']}_{ws_name}_{ts}.csv"
#                         filepath = os.path.join(DATA_DIR, filename)
#                         norm.to_csv(filepath, index=False)
#                         cur.execute("INSERT INTO uploads (user_id,filename,filepath,uploaded_at,field_name,notes) VALUES (?,?,?,?,?,?)",
#                                     (user['id'], filename, filepath, datetime.utcnow().isoformat(), ws_name, "manual workspace"))
#                         conn.commit()
#                         st.success(f"Workspace saved as {filename}")
# ############################################################################################################
# # ------------------ FORECAST ANALYSIS ------------------
# with tabs[1]:
#     st.header("Forecast Analysis")
#     st.write("Combine uploaded files (or pick a file) and run analysis. Deferments will zero production in selected windows and be shaded on plots.")

#     # # List user's uploads
#     # cur.execute("SELECT id,filename,filepath,field_name,uploaded_at FROM uploads WHERE user_id=? ORDER BY uploaded_at DESC", (user['id'],))
#     # myfiles = cur.fetchall()
#     # files_df = pd.DataFrame(myfiles, columns=['id','filename','filepath','field_name','uploaded_at']) if myfiles else pd.DataFrame()

#     files_df = pd.DataFrame()  # ✅ always define first

#     if user:  # <-- Only run if logged in
#         # List user's uploads
#         cur.execute("""
#             SELECT id,filename,filepath,field_name,uploaded_at 
#             FROM uploads 
#             WHERE user_id=? 
#             ORDER BY uploaded_at DESC
#         """,(user['id'],))

#         myfiles = cur.fetchall()
#         files_df = pd.DataFrame(myfiles, columns=['id','filename','filepath','field_name','uploaded_at']) if myfiles else pd.DataFrame()
  


#     if files_df.empty:
#         st.info("No uploads found. Use Production Input to upload or create a manual workspace.")
#     else:
#         sel_files = st.multiselect("Select files to include in analysis (multiple allowed)", files_df['filename'].tolist())
#         if sel_files:
#             dfs = []
#             for fn in sel_files:
#                 fp = files_df.loc[files_df['filename'] == fn, 'filepath'].values[0]
#                 if os.path.exists(fp):
#                     df = pd.read_csv(fp, parse_dates=['Date'])
#                     df['source_file'] = fn
#                     dfs.append(df)
#             if not dfs:
#                 st.error("Selected files not found on disk.")
#             else:
#                 big = pd.concat(dfs, ignore_index=True)
#                 big['Date'] = pd.to_datetime(big['Date'], errors='coerce')
#                 st.subheader("Combined preview")
#                 st.dataframe(big.head(50))

#                 freq = st.selectbox("Aggregation freq", ['D', 'M', 'Y'], index=1, help='D=Daily, M=Monthly, Y=Yearly')
#                 horizon_choice = st.selectbox("Forecast horizon unit", ['Days', 'Months', 'Years'], index=2)
#                 horizon_value = st.number_input("Horizon amount (integer)", min_value=1, max_value=100000, value=10)
#                 if st.button("Run analysis"):
#                     if freq == 'M':
#                         big['period'] = big['Date'].dt.to_period('M').dt.to_timestamp()
#                     elif freq == 'Y':
#                         big['period'] = big['Date'].dt.to_period('A').dt.to_timestamp()
#                     else:
#                         big['period'] = big['Date']

#                     agg = big.groupby('period')[['Oil (BOPD)', 'Gas (MMSCFD)']].sum().reset_index().rename(columns={'period':'Date'})
#                     agg['Date'] = pd.to_datetime(agg['Date'])
#                     st.session_state['agg_cache'] = agg.copy()  # cache for KNN section
#                     st.subheader('Aggregated series')

#                     # Apply deferments to aggregated series for visualization
#                     deferments = st.session_state.get("deferments", {})
#                     agg_adj = apply_deferments(agg, deferments)

#                     # Apply scheduled ramp-ups/declines to aggregated (historical) series for visualization
#                     # Use st.session_state ramp/decline dicts directly; function will interpret the dict structure
#                     agg_adj = apply_scheduled_changes(agg_adj, st.session_state.get("rampups", {}), "Oil (BOPD)")
#                     agg_adj = apply_scheduled_changes(agg_adj, st.session_state.get("declines", {}), "Oil (BOPD)")
#                     agg_adj = apply_scheduled_changes(agg_adj, st.session_state.get("rampups", {}), "Gas (MMSCFD)")
#                     agg_adj = apply_scheduled_changes(agg_adj, st.session_state.get("declines", {}), "Gas (MMSCFD)")

#                     # The target average stored (if any) should be applied optionally for display
#                     ta = st.session_state.get("target_avg", {})
#                     if ta:
#                         if ta.get("Oil"):
#                             agg_adj = apply_target_average(agg_adj, target_avg=ta.get("Oil"), column="Oil (BOPD)", lock_schedules=ta.get("lock", True))
#                         if ta.get("Gas"):
#                             agg_adj = apply_target_average(agg_adj, target_avg=ta.get("Gas"), column="Gas (MMSCFD)", lock_schedules=ta.get("lock", True))

#                     st.line_chart(agg_adj.set_index('Date')[['Oil (BOPD)', 'Gas (MMSCFD)']])

#                     # Matplotlib plot with shading (historical)
#                     fig, ax = plt.subplots(figsize=(10,4))
#                     ax.plot(agg_adj['Date'], agg_adj['Oil (BOPD)'], label='Oil (BOPD)')
#                     ax.plot(agg_adj['Date'], agg_adj['Gas (MMSCFD)'], label='Gas (MMSCFD)')
#                     shade_deferment_spans(ax, deferments)
#                     ax.set_xlabel('Date'); ax.set_ylabel('Production')
#                     ax.legend(); ax.grid(True)
#                     st.pyplot(fig)

#     # --- KNN extended forecasting (re-usable) ---
#     st.markdown("---")
#     st.subheader("KNN-based Extended Forecast")
#     knn_source_df = st.session_state.get("agg_cache", None)
#     if knn_source_df is None and not files_df.empty:
#         pick = st.selectbox("Pick single uploaded file for KNN (if no agg cached)", files_df['filename'].tolist(), key='knn_pick')
#         if pick:
#             fp = files_df.loc[files_df['filename']==pick,'filepath'].values[0]
#             if os.path.exists(fp):
#                 tmp = pd.read_csv(fp, parse_dates=['Date'])
#                 tmp = tmp.groupby('Date')[['Oil (BOPD)','Gas (MMSCFD)']].sum().reset_index()
#                 knn_source_df = tmp.sort_values('Date')
#     if knn_source_df is None or knn_source_df.empty:
#         st.info("No data available for KNN. Create/upload and aggregate first.")
#     else:
#         series_choice = st.radio("Model series", ["Oil (BOPD)", "Gas (MMSCFD)"], horizontal=True)
#         df_prod = knn_source_df[['Date', series_choice]].rename(columns={series_choice:'Production'})

#         # Apply deferments to the series
#         df_prod = apply_deferments(df_prod, st.session_state.get("deferments", {}))

#         if df_prod.empty:
#             st.warning("No data points to model.")
#         else:
#             if not SKLEARN_AVAILABLE:
#                 st.error("scikit-learn not available. `pip install scikit-learn` to enable KNN forecasting.")
#             else:
#                 with st.expander("KNN Settings"):
#                     max_val = int(df_prod['Production'].max() * 2) if df_prod['Production'].max() > 0 else 10000
#                     target_avg = st.slider("Target Average (BOPD/MMscfd)", 0, max_val, min(4500, max_val//2), 100)
#                     n_neighbors = st.slider("KNN neighbors", 1, 20, 3)
#                     extend_years = st.slider("Forecast horizon (years)", 1, 75, 10)

#                 df_prod = df_prod.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
#                 lock_end = df_prod['Date'].max()
#                 hist = df_prod.copy()
#                 hist['Days'] = (hist['Date'] - hist['Date'].min()).dt.days
#                 X = hist[['Days']]
#                 y = hist['Production'].fillna(0)
#                 knn = KNeighborsRegressor(n_neighbors=n_neighbors)
#                 knn.fit(X, y)

#                 future_end = lock_end + pd.DateOffset(years=int(extend_years))
#                 future_days = pd.date_range(start=lock_end + pd.Timedelta(days=1), end=future_end, freq='D')
#                 future_X = (future_days - hist['Date'].min()).days.values.reshape(-1,1)
#                 future_pred = knn.predict(future_X)
#                 future_df = pd.DataFrame({'Date': future_days, 'Production': future_pred})
#                 forecast_df = pd.concat([hist[['Date','Production']], future_df], ignore_index=True).reset_index(drop=True)

#                 # *** APPLY SCHEDULES TO KNN FORECAST & HISTORICAL PRODUCTION ***
#                 # Our schedules in session_state are dicts like {'ramp_1': {'date': 'YYYY-MM-DD', 'rate': 5000}, ...}
#                 # apply_scheduled_changes can accept these dicts and will compute absolute targets or convert pct declines.
#                 # Apply to historical part first (for display / editing)
#                 hist_part = forecast_df[forecast_df['Date'] <= lock_end].reset_index(drop=True)
#                 hist_part = apply_scheduled_changes(hist_part, st.session_state.get("rampups", {}), "Production")
#                 hist_part = apply_scheduled_changes(hist_part, st.session_state.get("declines", {}), "Production")

#                 # Then apply to forecast part
#                 fcst_part = forecast_df[forecast_df['Date'] > lock_end].reset_index(drop=True)
#                 # When schedules reference dates beyond history, apply_scheduled_changes will create/insert rows and interpolate
#                 fcst_part = apply_scheduled_changes(fcst_part, st.session_state.get("rampups", {}), "Production")
#                 fcst_part = apply_scheduled_changes(fcst_part, st.session_state.get("declines", {}), "Production")

#                 # Recombine and ensure ordering
#                 forecast_df = pd.concat([hist_part, fcst_part], ignore_index=True).sort_values('Date').reset_index(drop=True)

#                 # Apply deferments to forecast (set to 0)
#                 deferments = st.session_state.get("deferments", {})
#                 defer_dates = []
#                 for d in deferments.values():
#                     if d.get('reason') and d.get('reason') != 'None' and int(d.get('duration_days',0))>0:
#                         sd = pd.to_datetime(d.get('start_date'), errors='coerce')
#                         if pd.isna(sd):
#                             continue
#                         days = int(d.get('duration_days',0))
#                         defer_dates += pd.date_range(sd, periods=days, freq='D').tolist()
#                 if defer_dates:
#                     forecast_df['Deferred'] = forecast_df['Date'].isin(defer_dates)
#                     forecast_df.loc[forecast_df['Deferred'], 'Production'] = 0
#                 else:
#                     forecast_df['Deferred'] = False

#                 # Rescale future non-deferred days to hit target_avg (over whole forecast_df)
#                 total_days = len(forecast_df)
#                 if total_days>0:
#                     hist_mask = forecast_df['Date'] <= lock_end
#                     hist_cum = forecast_df.loc[hist_mask, 'Production'].sum()
#                     required_total = target_avg * total_days
#                     required_future_prod = required_total - hist_cum
#                     valid_future_mask = (forecast_df['Date'] > lock_end) & (~forecast_df['Deferred'])
#                     num_valid = valid_future_mask.sum()
#                     if num_valid > 0:
#                         new_avg = required_future_prod / num_valid
#                         forecast_df.loc[valid_future_mask, 'Production'] = new_avg

#                 # Now apply global target average if set in sidebar session (this keeps original behavior)
#                 ta = st.session_state.get("target_avg", {})
#                 if ta and ta.get("Oil") and series_choice == "Oil (BOPD)":
#                     forecast_df = apply_target_average(forecast_df, target_avg=ta.get("Oil"), column="Production", lock_schedules=ta.get("lock", True))
#                 if ta and ta.get("Gas") and series_choice == "Gas (MMSCFD)":
#                     forecast_df = apply_target_average(forecast_df, target_avg=ta.get("Gas"), column="Production", lock_schedules=ta.get("lock", True))

#                 forecast_df['Year'] = forecast_df['Date'].dt.year
#                 min_year = int(forecast_df['Year'].min())
#                 max_year = int(forecast_df['Year'].max())
#                 sel_years = st.slider("Select Year Range", min_year, max_year, (min_year, max_year))
#                 analysis_df = forecast_df[(forecast_df['Year'] >= sel_years[0]) & (forecast_df['Year'] <= sel_years[1])]

#                 # metrics & plot
#                 st.metric("Cumulative Production", f"{analysis_df['Production'].sum():,.0f}")
#                 fig, ax = plt.subplots(figsize=(12,4))
#                 ax.plot(analysis_df['Date'], analysis_df['Production'], label='Production', color="green")
#                 ax.axhline(target_avg, linestyle='--', label='Target Avg', color="red")
#                 ax.axvline(lock_end, linestyle='--', label='End of History', color="black")
#                 shade_deferment_spans(ax, deferments)
#                 ax.set_title(f"KNN Forecast — {series_choice}")
#                 ax.set_xlabel("Date"); ax.set_ylabel(series_choice)
#                 ax.legend(); ax.grid(True)
#                 st.pyplot(fig)

#                 # editable tables
#                 st.subheader("Edit Historical")
#                 historical_data = analysis_df[analysis_df['Date'] <= lock_end][['Date','Production','Deferred']]
#                 hist_edit = st.data_editor(historical_data, num_rows='dynamic', key='knn_hist_editor')
#                 st.subheader("Edit Forecast")
#                 forecast_only = analysis_df[analysis_df['Date'] > lock_end][['Date','Production','Deferred']]
#                 forecast_only = forecast_only[~forecast_only['Date'].isin(hist_edit['Date'])]
#                 fcst_edit = st.data_editor(forecast_only, num_rows='dynamic', key='knn_fcst_editor')

#                 merged = pd.concat([hist_edit, fcst_edit], ignore_index=True).sort_values('Date')
#                 st.subheader("Forecast Data (editable)")
#                 st.dataframe(merged, hide_index=True)

#                 # downloads
#                 csv_data = merged.to_csv(index=False).encode('utf-8')
#                 excel_buf = BytesIO()
#                 with pd.ExcelWriter(excel_buf, engine='xlsxwriter') as writer:
#                     merged.to_excel(writer, sheet_name='Forecast', index=False)
#                 st.download_button("Download CSV", data=csv_data, file_name='forecast.csv')
#                 st.download_button("Download Excel", data=excel_buf.getvalue(), file_name='forecast.xlsx')

# # # ------------------ ACCOUNT & ADMIN ------------------
# # with tabs[2]:
# #     st.header("Account & Admin")
# #     st.subheader("Account info")
# #     st.write(user)
# #     st.markdown("---")
# #     # if user['is_admin']:
# #     if user and user.get('is_admin'):

# #         st.subheader("Admin: view all uploads")
# #         cur.execute("SELECT u.id,u.username,a.filename,a.uploaded_at,a.field_name,a.filepath FROM users u JOIN uploads a ON u.id=a.user_id ORDER BY a.uploaded_at DESC LIMIT 500")
# #         allrows = cur.fetchall()
# #         if allrows:
# #             adf = pd.DataFrame(allrows, columns=['user_id','username','filename','uploaded_at','field_name','filepath'])
# #             st.dataframe(adf)
# #             if st.button("Download combined CSV of all uploads"):
# #                 frames = []
# #                 for fp in adf['filepath']:
# #                     if os.path.exists(fp):
# #                         frames.append(pd.read_csv(fp))
# #                 if frames:
# #                     combined = pd.concat(frames, ignore_index=True)
# #                     buf = BytesIO()
# #                     combined.to_csv(buf, index=False)
# #                     buf.seek(0)
# #                     st.download_button("Download combined CSV (final)", data=buf, file_name='combined_all_users.csv')
# #                 else:
# #                     st.warning("No files found on disk")
# #         else:
# #             st.info("No uploads yet")
# #     else:
# #         st.info("You are not an admin. Admins can view/download uploads.")

# # st.sidebar.markdown('---')
# # st.sidebar.write(f"Local DB: {DB_PATH}")

# # import os

# # # Directory where files are stored
# # DATA_DIR = os.path.join(os.getcwd(), "uploaded_files")
# #######################################
# # # ------------------ ACCOUNT & ADMIN ------------------
# # with tabs[2]:
# #     st.header("Account & Admin")

# #     st.subheader("Account info")
# #     if user:
# #         st.json(user)
# #     else:
# #         st.warning("⚠️ No user logged in")
# #     st.markdown("---")

# #     # Admin features
# #     if user and user.get('is_admin'):
# #         st.subheader("Admin: view all uploads")
# #         cur.execute("""
# #             SELECT u.id as user_id, u.username, a.filename, a.uploaded_at, a.field_name, a.filepath
# #             FROM users u 
# #             JOIN uploads a ON u.id = a.user_id 
# #             ORDER BY a.uploaded_at DESC 
# #             LIMIT 500
# #         """)
# #         allrows = cur.fetchall()

# #         if allrows:
# #             adf = pd.DataFrame(allrows, columns=['user_id','username','filename','uploaded_at','field_name','filepath'])
# #             st.dataframe(adf)

# #             if st.button("Download combined CSV of all uploads"):
# #                 frames = []
# #                 for fp in adf['filepath']:
# #                     if os.path.exists(fp):
# #                         frames.append(pd.read_csv(fp))
# #                 if frames:
# #                     combined = pd.concat(frames, ignore_index=True)
# #                     buf = BytesIO()
# #                     combined.to_csv(buf, index=False)
# #                     buf.seek(0)
# #                     st.download_button(
# #                         "Download combined CSV (final)",
# #                         data=buf,
# #                         file_name='combined_all_users.csv'
# #                     )
# #                 else:
# #                     st.warning("No files found on disk")
# #         else:
# #             st.info("No uploads yet")
# #     else:
# #         st.info("You are not an admin. Admins can view/download uploads.")

# # st.sidebar.markdown('---')
# # if user and user.get("is_admin"):
# #     st.sidebar.write(f"Local DB: {DB_PATH}")

# # # Directory where files are stored
# # DATA_DIR = os.path.join(os.getcwd(), "uploaded_files")
# #######################################
# import logging

# # ------------------ LOGGING SETUP ------------------
# LOG_FILE = "app_errors.log"
# logging.basicConfig(
#     filename=LOG_FILE,
#     level=logging.ERROR,
#     format="%(asctime)s [%(levelname)s] %(message)s"
# )

# # ------------------ ACCOUNT & ADMIN ------------------
# with tabs[2]:
#     st.header("Account & Admin")

#     st.subheader("Account info")
#     if user:
#         st.json(user)
#     else:
#         st.warning("⚠️ No user logged in")
#     st.markdown("---")

#     # Admin features
#     if user and user.get('is_admin'):
#         st.subheader("🔑 Admin: Manage All Uploads")

#         try:
#             # Get all users
#             cur.execute("SELECT id, username FROM users ORDER BY username ASC")
#             all_users = cur.fetchall()
#         except Exception as e:
#             logging.error(f"Failed to fetch users: {e}", exc_info=True)
#             all_users = []

#         if all_users:
#             # User filter dropdown
#             usernames = {u[1]: u[0] for u in all_users}
#             selected_user = st.selectbox("👤 Select a user", list(usernames.keys()))

#             # Fetch that user’s uploads
#             cur.execute(
#                 "SELECT filename, uploaded_at, field_name, filepath FROM uploads WHERE user_id=? ORDER BY uploaded_at DESC",
#                 (usernames[selected_user],)
#             )
#             user_uploads = cur.fetchall()

#             if user_uploads:
#                 df_user = pd.DataFrame(user_uploads, columns=['Filename', 'Uploaded At', 'Field', 'Filepath'])
#                 st.dataframe(df_user[['Filename', 'Uploaded At', 'Field']])

#                 # Pick a file
#                 file_choice = st.selectbox("📂 Select a file to view", df_user['Filename'])

#                 # Show preview or allow download
#                 chosen_row = df_user[df_user['Filename'] == file_choice].iloc[0]
#                 filepath = chosen_row['Filepath']

#                 if os.path.exists(filepath):
#                     try:
#                         df_preview = pd.read_csv(filepath)
#                         st.write("📊 Preview of selected file:")
#                         st.dataframe(df_preview.head(20))  # show first 20 rows

#                         # Allow download
#                         with open(filepath, "rb") as f:
#                             st.download_button(
#                                 label="⬇️ Download file",
#                                 data=f,
#                                 file_name=chosen_row['Filename'],
#                                 mime="text/csv"
#                             )
#                     except Exception as e:
#                         logging.error(f"Failed to read file {filepath}: {e}", exc_info=True)
#                         st.error("⚠️ Could not open this file.")
#                 else:
#                     st.error("⚠️ File not found on disk.")
#             else:
#                 st.info("This user has no uploads.")
#         else:
#             st.info("No users found in system.")



# #################
# # # ------------------ ACCOUNT & ADMIN ------------------
# # with tabs[2]:
# #     st.header("Account & Admin")

# #     st.subheader("Account info")
# #     if user:
# #         st.json(user)
# #     else:
# #         st.warning("⚠️ No user logged in")
# #     st.markdown("---")

# #     # Admin features
# #     if user and user.get('is_admin'):
# #         st.subheader("Admin: view all uploads")

# #         try:
# #             cur.execute("""
# #                 SELECT u.id as user_id, u.username, a.filename, a.uploaded_at, a.field_name, a.filepath
# #                 FROM users u 
# #                 JOIN uploads a ON u.id = a.user_id 
# #                 ORDER BY a.uploaded_at DESC 
# #                 LIMIT 500
# #             """)
# #             allrows = cur.fetchall()
# #         except Exception as e:
# #             logging.error(f"Database query failed: {e}", exc_info=True)
# #             st.error("⚠️ Could not load uploads. Please try again later.")
# #             allrows = []

# #         if allrows:
# #             adf = pd.DataFrame(allrows, columns=['user_id','username','filename','uploaded_at','field_name','filepath'])
# #             st.dataframe(adf)

# #             if st.button("Download combined CSV of all uploads"):
# #                 frames = []
# #                 for fp in adf['filepath']:
# #                     if os.path.exists(fp):
# #                         try:
# #                             frames.append(pd.read_csv(fp))
# #                         except Exception as e:
# #                             logging.error(f"Failed to read file {fp}: {e}", exc_info=True)
# #                             st.warning(f"⚠️ Skipped a corrupted/missing file.")
# #                 if frames:
# #                     combined = pd.concat(frames, ignore_index=True)
# #                     buf = BytesIO()
# #                     combined.to_csv(buf, index=False)
# #                     buf.seek(0)
# #                     st.download_button(
# #                         "Download combined CSV (final)",
# #                         data=buf,
# #                         file_name='combined_all_users.csv'
# #                     )
# #                 else:
# #                     st.warning("No files found on disk")
# #         else:
# #             st.info("No uploads yet")
# #     else:
# #         st.info("You are not an admin. Admins can view/download uploads.")

# # st.sidebar.markdown('---')
# # if user and user.get("is_admin"):
# #     st.sidebar.write(f"Local DB: {DB_PATH}")

# # # Directory where files are stored
# # DATA_DIR = os.path.join(os.getcwd(), "uploaded_files")




# # # ------------------ RECENT UPLOADS TAB ------------------
# # with tabs[3]:
# #     st.header("🕒 Recent Uploads")

# #     if user["is_admin"]:
# #         cur.execute("""
# #             SELECT u.filename, u.field_name, u.notes, u.uploaded_at, usr.username
# #             FROM uploads u
# #             JOIN users usr ON u.user_id = usr.id
# #             ORDER BY u.uploaded_at DESC
# #             LIMIT 10
# #         """)
# #         rows = cur.fetchall()

# #         if rows:
# #             st.subheader("All Users' Recent Uploads (Admin)")
# #             for idx, r in enumerate(rows):
# #                 col1, col2, col3, col4 = st.columns([2,3,2,2])
# #                 col1.write(r[0])   # filename
# #                 col2.write(r[4])   # uploaded_by
# #                 col3.write(r[3])   # uploaded_at

# #                 # Use separate keys
# #                 confirm_state_key = f"admin_confirm_state_{idx}"
# #                 delete_button_key = f"admin_delete_btn_{idx}"
# #                 confirm_button_key = f"admin_confirm_btn_{idx}"

# #                 if st.session_state.get(confirm_state_key, False):
# #                     if col4.button("❌ Confirm Delete", key=confirm_button_key):
# #                         # Delete from DB
# #                         cur.execute("DELETE FROM uploads WHERE filename=?", (r[0],))
# #                         conn.commit()

# #                         # Delete from disk
# #                         file_path = os.path.join(DATA_DIR, r[0])
# #                         if os.path.exists(file_path):
# #                             os.remove(file_path)

# #                         st.warning(f"Admin deleted {r[0]}")
# #                         st.session_state[confirm_state_key] = False
# #                         st.experimental_rerun()
# #                 else:
# #                     if col4.button("🗑️ Delete", key=delete_button_key):
# #                         st.session_state[confirm_state_key] = True
# #                         st.error(f"⚠️ Click confirm to delete {r[0]}")
# #         else:
# #             st.info("No recent uploads in the system.")

# #     else:
# #         cur.execute(
# #             "SELECT filename, field_name, notes, uploaded_at FROM uploads WHERE user_id = ? ORDER BY uploaded_at DESC LIMIT 5", 
# #             (user['id'],) if user else (-1,)

# #         )
# #         rows = cur.fetchall()

# #         if rows:
# #             st.subheader("My Recent Uploads")
# #             for idx, r in enumerate(rows):
# #                 col1, col2, col3 = st.columns([3,3,2])
# #                 col1.write(r[0])   # filename
# #                 col2.write(r[3])   # uploaded_at

# #                 # Use separate keys
# #                 confirm_state_key = f"user_confirm_state_{idx}"
# #                 delete_button_key = f"user_delete_btn_{idx}"
# #                 confirm_button_key = f"user_confirm_btn_{idx}"

# #                 if st.session_state.get(confirm_state_key, False):
# #                     if col3.button("❌ Confirm Delete", key=confirm_button_key):
# #                         # Delete from DB
# #                         cur.execute("DELETE FROM uploads WHERE filename=? AND user_id=?", (r[0], user['id']))
# #                         conn.commit()

# #                         # Delete from disk
# #                         file_path = os.path.join(DATA_DIR, r[0])
# #                         if os.path.exists(file_path):
# #                             os.remove(file_path)

# #                         st.success(f"Deleted {r[0]}")
# #                         st.session_state[confirm_state_key] = False
# #                         st.experimental_rerun()
# #                 else:
# #                     if col3.button("🗑️ Delete", key=delete_button_key):
# #                         st.session_state[confirm_state_key] = True
# #                         st.error(f"⚠️ Click confirm to delete {r[0]}")
# #         else:
# #             st.info("You have no recent uploads.")
# #################################################################
# # ------------------ RECENT UPLOADS TAB ------------------
# with tabs[3]:
#     st.header("🕒 Recent Uploads")

#     if user:  # ✅ only run if someone is logged in
#         if user.get("is_admin"):
#             cur.execute("""
#                 SELECT u.filename, u.field_name, u.notes, u.uploaded_at, usr.username
#                 FROM uploads u
#                 JOIN users usr ON u.user_id = usr.id
#                 ORDER BY u.uploaded_at DESC
#                 LIMIT 10
#             """)
#             rows = cur.fetchall()

#             if rows:
#                 st.subheader("All Users' Recent Uploads (Admin)")
#                 for idx, r in enumerate(rows):
#                     col1, col2, col3, col4 = st.columns([2,3,2,2])
#                     col1.write(r[0])   # filename
#                     col2.write(r[4])   # uploaded_by
#                     col3.write(r[3])   # uploaded_at

#                     # Use separate keys
#                     confirm_state_key = f"admin_confirm_state_{idx}"
#                     delete_button_key = f"admin_delete_btn_{idx}"
#                     confirm_button_key = f"admin_confirm_btn_{idx}"

#                     if st.session_state.get(confirm_state_key, False):
#                         if col4.button("❌ Confirm Delete", key=confirm_button_key):
#                             # Delete from DB
#                             cur.execute("DELETE FROM uploads WHERE filename=?", (r[0],))
#                             conn.commit()

#                             # Delete from disk
#                             file_path = os.path.join(DATA_DIR, r[0])
#                             if os.path.exists(file_path):
#                                 os.remove(file_path)

#                             st.warning(f"Admin deleted {r[0]}")
#                             st.session_state[confirm_state_key] = False
#                             st.experimental_rerun()
#                     else:
#                         if col4.button("🗑️ Delete", key=delete_button_key):
#                             st.session_state[confirm_state_key] = True
#                             st.error(f"⚠️ Click confirm to delete {r[0]}")
#             else:
#                 st.info("No recent uploads in the system.")

#         else:  # normal user
#             cur.execute(
#                 "SELECT filename, field_name, notes, uploaded_at FROM uploads WHERE user_id = ? ORDER BY uploaded_at DESC LIMIT 5", 
#                 (user['id'],)
#             )
#             rows = cur.fetchall()

#             if rows:
#                 st.subheader("My Recent Uploads")
#                 for idx, r in enumerate(rows):
#                     col1, col2, col3 = st.columns([3,3,2])
#                     col1.write(r[0])   # filename
#                     col2.write(r[3])   # uploaded_at

#                     # Use separate keys
#                     confirm_state_key = f"user_confirm_state_{idx}"
#                     delete_button_key = f"user_delete_btn_{idx}"
#                     confirm_button_key = f"user_confirm_btn_{idx}"

#                     if st.session_state.get(confirm_state_key, False):
#                         if col3.button("❌ Confirm Delete", key=confirm_button_key):
#                             # Delete from DB
#                             cur.execute("DELETE FROM uploads WHERE filename=? AND user_id=?", (r[0], user['id']))
#                             conn.commit()

#                             # Delete from disk
#                             file_path = os.path.join(DATA_DIR, r[0])
#                             if os.path.exists(file_path):
#                                 os.remove(file_path)

#                             st.success(f"Deleted {r[0]}")
#                             st.session_state[confirm_state_key] = False
#                             st.experimental_rerun()
#                     else:
#                         if col3.button("🗑️ Delete", key=delete_button_key):
#                             st.session_state[confirm_state_key] = True
#                             st.error(f"⚠️ Click confirm to delete {r[0]}")
#             else:
#                 st.info("You have no recent uploads.")
#     else:
#         st.info("Please log in to view recent uploads.")  # ✅ safe when not logged in





# # # ------------------ SAVED FILES TAB ------------------
# # with tabs[4]:  # adjust index depending on your layout query language
# #     st.header("📂 Saved Files")

# #     if user["is_admin"]:
# #         cur.execute("""
# #             SELECT u.filepath, u.filename, u.field_name, u.notes, u.uploaded_at, usr.username
# #             FROM uploads u
# #             JOIN users usr ON u.user_id = usr.id
# #             ORDER BY u.uploaded_at DESC
# #         """)
# #         rows = cur.fetchall()

# #         if rows:
# #             df_files = pd.DataFrame(rows, columns=["Filepath", "Filename", "Field", "Notes", "Uploaded At", "Uploaded By"])
# #             for idx, row in df_files.iterrows():
# #                 st.markdown(f"**{row['Filename']}** — Field: {row['Field']} | By: {row['Uploaded By']} | {row['Uploaded At']}")
# #                 try:
# #                     with open(row["Filepath"], "rb") as f:
# #                         st.download_button(
# #                             label="⬇️ Download",
# #                             data=f,
# #                             file_name=row["Filename"],
# #                             mime="text/csv",
# #                             key=f"download_{idx}"
# #                         )
# #                 except FileNotFoundError:
# #                     st.error(f"File {row['Filename']} not found on disk.")
# #         else:
# #             st.info("No files saved in the system yet.")
# #     else:
# #         cur.execute(
# #             "SELECT filepath, filename, field_name, notes, uploaded_at FROM uploads WHERE user_id = ? ORDER BY uploaded_at DESC",
# #             (user['id'],) if user else (-1,)

# #         )
# #         rows = cur.fetchall()

# #         if rows:
# #             df_files = pd.DataFrame(rows, columns=["Filepath", "Filename", "Field", "Notes", "Uploaded At"])
# #             for idx, row in df_files.iterrows():
# #                 st.markdown(f"**{row['Filename']}** — Field: {row['Field']} | {row['Uploaded At']}")
# #                 try:
# #                     with open(row["Filepath"], "rb") as f:
# #                         st.download_button(
# #                             label="⬇️ Download",
# #                             data=f,
# #                             file_name=row["Filename"],
# #                             mime="text/csv",
# #                             key=f"user_download_{idx}"
# #                         )
# #                 except FileNotFoundError:
# #                     st.error(f"File {row['Filename']} not found on disk.")
# #         else:
# #             st.info("You have not saved any files yet.")
# #############################
# # ------------------ SAVED FILES TAB ------------------
# with tabs[4]:  # adjust index depending on your layout
#     st.header("📂 Saved Files")

#     if user:  # ✅ only run if logged in
#         if user.get("is_admin"):
#             cur.execute("""
#                 SELECT u.filepath, u.filename, u.field_name, u.notes, u.uploaded_at, usr.username
#                 FROM uploads u
#                 JOIN users usr ON u.user_id = usr.id
#                 ORDER BY u.uploaded_at DESC
#             """)
#             rows = cur.fetchall()

#             if rows:
#                 df_files = pd.DataFrame(
#                     rows, 
#                     columns=["Filepath", "Filename", "Field", "Notes", "Uploaded At", "Uploaded By"]
#                 )
#                 for idx, row in df_files.iterrows():
#                     st.markdown(f"**{row['Filename']}** — Field: {row['Field']} | By: {row['Uploaded By']} | {row['Uploaded At']}")
#                     try:
#                         with open(row["Filepath"], "rb") as f:
#                             st.download_button(
#                                 label="⬇️ Download",
#                                 data=f,
#                                 file_name=row["Filename"],
#                                 mime="text/csv",
#                                 key=f"download_{idx}"
#                             )
#                     except FileNotFoundError:
#                         st.error(f"File {row['Filename']} not found on disk.")
#             else:
#                 st.info("No files saved in the system yet.")
#         else:  # normal user
#             cur.execute(
#                 "SELECT filepath, filename, field_name, notes, uploaded_at FROM uploads WHERE user_id = ? ORDER BY uploaded_at DESC",
#                 (user['id'],)
#             )
#             rows = cur.fetchall()

#             if rows:
#                 df_files = pd.DataFrame(
#                     rows, 
#                     columns=["Filepath", "Filename", "Field", "Notes", "Uploaded At", "Uploaded By"]
#                 )
#                 for idx, row in df_files.iterrows():
#                     st.markdown(f"**{row['Filename']}** — Field: {row['Field']} | {row['Uploaded At']}")
#                     try:
#                         with open(row["Filepath"], "rb") as f:
#                             st.download_button(
#                                 label="⬇️ Download",
#                                 data=f,
#                                 file_name=row["Filename"],
#                                 mime="text/csv",
#                                 key=f"user_download_{idx}"
#                             )
#                     except FileNotFoundError:
#                         st.error(f"File {row['Filename']} not found on disk.")
#             else:
#                 st.info("You have not saved any files yet.")
#     else:
#         st.info("Please log in to view saved files.")  # ✅ safe when not logged in






# ###################################################################################################
# ###################################################################################
# #####################################################################
# # production_manager_full.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import sqlite3
# import os
# import hashlib
# import time
# from datetime import datetime, timedelta
# from io import BytesIO
# import matplotlib.pyplot as plt


# # optional: KNN
# try:
#     from sklearn.neighbors import KNeighborsRegressor
#     SKLEARN_AVAILABLE = True
# except Exception:
#     SKLEARN_AVAILABLE = False

# # ------------------ CONFIG ------------------
# DB_PATH = "production_app.db"
# DATA_DIR = "data_uploads"
# os.makedirs(DATA_DIR, exist_ok=True)

# # ------------------ DB ------------------
# conn = sqlite3.connect(DB_PATH, check_same_thread=False)
# cur = conn.cursor()


# # # ------------------ DB ------------------
# # conn = sqlite3.connect(DB_PATH, check_same_thread=False)
# # cur = conn.cursor()


# # import sqlite3

# # DB_PATH = "production_app.db"

# # def reset_users_table():
# #     # Use a separate connection just for this operation
# #     with sqlite3.connect(DB_PATH) as conn:
# #         cur = conn.cursor()

# #         # Drop the table if it exists
# #         cur.execute("DROP TABLE IF EXISTS users")

# #         # Recreate the table
# #         cur.execute('''
# #         CREATE TABLE users (
# #             id INTEGER PRIMARY KEY AUTOINCREMENT,
# #             username TEXT UNIQUE,
# #             email TEXT UNIQUE,
# #             password_hash TEXT,
# #             is_admin INTEGER DEFAULT 0,
# #             is_verified INTEGER DEFAULT 0,
# #             verify_token TEXT,
# #             reset_token TEXT,
# #             created_at TEXT
# #         )
# #         ''')

# #         conn.commit()
# #         print("✅ Users table has been reset successfully!")

# # # Call this function once when you want to reset
# # reset_users_table()

# # --- Create users table if it doesn't exist ---
# cur.execute('''
# CREATE TABLE IF NOT EXISTS users (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     username TEXT UNIQUE,
#     email TEXT UNIQUE,
#     password_hash TEXT,
#     is_admin INTEGER DEFAULT 0,
#     created_at TEXT
# )
# ''')

# # --- Create uploads table if it doesn't exist ---
# cur.execute('''
# CREATE TABLE IF NOT EXISTS uploads (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     user_id INTEGER,
#     filename TEXT,
#     filepath TEXT,
#     uploaded_at TEXT,
#     field_name TEXT,
#     notes TEXT,
#     source TEXT DEFAULT 'upload',
#     FOREIGN KEY(user_id) REFERENCES users(id)
# )
# ''')

# conn.commit()

# # --- Ensure the users table has the additional columns ---
# def ensure_user_columns():
#     cur.execute("PRAGMA table_info(users)")
#     existing = [row[1] for row in cur.fetchall()]

#     if "is_verified" not in existing:
#         cur.execute("ALTER TABLE users ADD COLUMN is_verified INTEGER DEFAULT 0")
#     if "verify_token" not in existing:
#         cur.execute("ALTER TABLE users ADD COLUMN verify_token TEXT")
#     if "reset_token" not in existing:
#         cur.execute("ALTER TABLE users ADD COLUMN reset_token TEXT")
#     conn.commit()

# ensure_user_columns()

# # ------------------ HELPERS & NEW SCHEDULING LOGIC ------------------
# REQUIRED_COLS = ["Date", "Oil (BOPD)", "Gas (MMSCFD)"]

# def hash_password(password: str) -> str:
#     salt = "static_salt_change_me"  # replace in production
#     return hashlib.sha256((salt + password).encode()).hexdigest()

# def create_user(username, email, password, is_admin=0):
#     ph = hash_password(password)
#     try:
#         cur.execute("INSERT INTO users (username,email,password_hash,is_admin,created_at) VALUES (?,?,?,?,?)",
#                     (username, email, ph, int(is_admin), datetime.utcnow().isoformat()))
#         conn.commit()
#         return True, "user created"
#     except sqlite3.IntegrityError as e:
#         return False, str(e)


# def authenticate(username_or_email, password):
#     ph = hash_password(password)
#     cur.execute(
#         "SELECT id, username, email, is_admin, is_verified FROM users WHERE (username=? OR email=?) AND password_hash=?",
#         (username_or_email, username_or_email, ph)
#     )
#     row = cur.fetchone()


#     cur.execute("SELECT id, username, email, is_verified FROM users")
#     print(cur.fetchall())

#     if row:
#         user = {
#             "id": row[0],
#             "username": row[1],
#             "email": row[2],
#             "is_admin": row[3],
#             "is_verified": row[4]  # ✅ include this!
#         }
#         return True, user
#     else:
#         return False, None

# def validate_and_normalize_df(df: pd.DataFrame):
#     # Normalize column names and check required
#     colmap = {}
#     for c in df.columns:
#         c_clean = c.strip()
#         if c_clean.lower() in [x.lower() for x in REQUIRED_COLS]:
#             for req in REQUIRED_COLS:
#                 if c_clean.lower() == req.lower():
#                     colmap[c] = req
#         elif 'date' in c_clean.lower():
#             colmap[c] = 'Date'
#         elif 'oil' in c_clean.lower():
#             colmap[c] = 'Oil (BOPD)'
#         elif 'gas' in c_clean.lower():
#             colmap[c] = 'Gas (MMSCFD)'
#         else:
#             colmap[c] = c_clean
#     df = df.rename(columns=colmap)
#     missing = [c for c in REQUIRED_COLS if c not in df.columns]
#     if missing:
#         return False, f"Missing required columns: {missing}"
#     try:
#         df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
#     except Exception as e:
#         return False, f"Date parse error: {e}"
#     if df['Date'].isna().any():
#         return False, "Some Date values could not be parsed. Ensure ISO-like dates (YYYY-MM-DD)."
#     for c in ['Oil (BOPD)','Gas (MMSCFD)']:
#         df[c] = pd.to_numeric(df[c], errors='coerce')
#         if df[c].isna().all():
#             return False, f"Column {c} has no numeric values"
#     return True, df

# def save_dataframe(user_id, df: pd.DataFrame, field_name: str, notes: str = "", source: str = "upload"):
#     ts = int(time.time())
#     safe_field = field_name.replace(" ", "_")[:60] if field_name else "unnamed"
#     filename = f"user{user_id}_{safe_field}_{ts}.csv"
#     filepath = os.path.join(DATA_DIR, filename)
#     df.to_csv(filepath, index=False)
#     cur.execute("INSERT INTO uploads (user_id,filename,filepath,uploaded_at,field_name,notes,source) VALUES (?,?,?,?,?,?,?)",
#                 (user_id, filename, filepath, datetime.utcnow().isoformat(), field_name, notes, source))
#     conn.commit()
#     return filename, filepath

# def apply_deferments(df, deferments):
#     """Apply deferments by zeroing Oil & Gas in deferment ranges."""
#     if df is None or df.empty or not deferments:
#         return df
#     df = df.copy()
#     if 'Date' not in df.columns:
#         return df
#     df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
#     for d in deferments.values():
#         reason = d.get('reason')
#         days = int(d.get('duration_days', 0) or 0)
#         start_str = d.get('start_date')
#         if reason and reason != "None" and days > 0 and start_str:
#             try:
#                 start = pd.to_datetime(start_str, errors='coerce')
#                 if pd.isna(start):
#                     continue
#                 end = start + pd.Timedelta(days=days-1)
#                 mask = (df['Date'] >= start) & (df['Date'] <= end)
#                 if 'Oil (BOPD)' in df.columns:
#                     df.loc[mask, 'Oil (BOPD)'] = 0
#                 if 'Gas (MMSCFD)' in df.columns:
#                     df.loc[mask, 'Gas (MMSCFD)'] = 0
#                 if 'Production' in df.columns:
#                     df.loc[mask, 'Production'] = 0
#             except Exception:
#                 continue
#     return df

# def shade_deferment_spans(ax, deferments, color_map=None):
#     spans = []
#     for d in deferments.values():
#         reason = d.get('reason')
#         days = int(d.get('duration_days', 0) or 0)
#         start_str = d.get('start_date')
#         if reason and reason != "None" and days > 0 and start_str:
#             s = pd.to_datetime(start_str, errors='coerce')
#             if pd.isna(s):
#                 continue
#             e = s + pd.Timedelta(days=days-1)
#             spans.append((s, e, reason))
#     for s, e, r in spans:
#         color = None
#         if color_map and r in color_map:
#             color = color_map[r]
#         ax.axvspan(s, e, alpha=0.18, color=color)

# # ------------------ NEW: SCHEDULING & TARGET AVERAGE HELPERS ------------------

# def _ensure_date_index(df):
#     """Return a copy indexed by Date (datetime), preserving original Date column too."""
#     d = df.copy()
#     if 'Date' in d.columns:
#         d['Date'] = pd.to_datetime(d['Date'], errors='coerce')
#         d = d.sort_values('Date').reset_index(drop=True)
#         d_indexed = d.set_index('Date', drop=False)
#     else:
#         # Try to use existing index
#         idx = pd.to_datetime(d.index)
#         d_indexed = d.copy()
#         d_indexed['Date'] = idx
#         d_indexed.index = idx
#     return d_indexed

# def apply_scheduled_changes(df, schedules, column):
#     """
#     Apply scheduled ramp-ups/declines with linear interpolation on the specified column.
#     schedules: dict where values are dicts (as stored in session_state) OR a mapping date->target
#     Accepts input df that has a 'Date' column or datetime index. Returns df with same shape.
#     """
#     if df is None or df.empty or not schedules or column not in df.columns:
#         return df

#     # Convert to DataFrame indexed by Date
#     df_i = _ensure_date_index(df)

#     # Accept two possible schedule formats:
#     # 1) state dict as in session_state: {'ramp_1': {'date':'YYYY-MM-DD','rate':5000}, ...}
#     # 2) direct mapping: {'2025-09-01': 5000, ...}
#     mapping = {}
#     # if schedules is a mapping of date->value (string/number)
#     if all((isinstance(k, (str, pd.Timestamp)) and not isinstance(v, dict)) for k, v in schedules.items()):
#         # treat as direct mapping
#         for k, v in schedules.items():
#             try:
#                 d = pd.to_datetime(k)
#                 mapping[d] = float(v)
#             except Exception:
#                 continue
#     else:
#         # treat as state dict
#         for entry in schedules.values():
#             try:
#                 d = pd.to_datetime(entry.get('date'))
#             except Exception:
#                 continue
#             if pd.isna(d):
#                 continue
#             # If entry has 'rate' it's absolute target
#             if 'rate' in entry:
#                 try:
#                     mapping[d] = float(entry.get('rate'))
#                 except Exception:
#                     continue
#             # If entry has 'pct' treat as a percent change to the base value at that date
#             elif 'pct' in entry:
#                 try:
#                     pct = float(entry.get('pct')) / 100.0
#                 except Exception:
#                     pct = 0.0
#                 # find base value just before or at that date
#                 if d in df_i.index:
#                     base = df_i.loc[d, column]
#                     if pd.isna(base):
#                         # take previous non-na
#                         prev = df_i.loc[:d, column].ffill().iloc[-1] if not df_i.loc[:d, column].ffill().empty else None
#                         base = prev if prev is not None else 0.0
#                 else:
#                     # take last known before date
#                     prev_series = df_i.loc[df_i.index <= d, column]
#                     if not prev_series.empty:
#                         base = prev_series.ffill().iloc[-1]
#                     else:
#                         # fallback: use earliest value
#                         base = df_i[column].ffill().iloc[0] if not df_i[column].ffill().empty else 0.0
#                 mapping[d] = float(base * (1 - pct))

#     if not mapping:
#         return df

#     # Keep only mapping dates within the df index range (we can still insert future dates if df extends into forecast)
#     # Insert scheduled targets (create rows if necessary)
#     for d, target in mapping.items():
#         if d not in df_i.index:
#             # if date outside range, add a new row so interpolation can happen
#             # create a new row using NaN for all columns, set Date index to d
#             new_row = pd.DataFrame([{c: np.nan for c in df_i.columns}])
#             new_row.index = pd.Index([d])
#             new_row['Date'] = d
#             df_i = pd.concat([df_i, new_row])
#     # Re-sort index
#     df_i = df_i.sort_index()

#     # Set scheduled absolute values
#     for d, target in mapping.items():
#         df_i.at[d, column] = target

#     # Interpolate the column across the entire index (linear)
#     df_i[column] = df_i[column].interpolate(method='time')  # time-aware interpolation

#     # Return df_i with original order and with Date column preserved
#     # Remove any rows we artificially added that were outside original df's full date coverage only if original had no such dates
#     # To be safe, convert back to same index shape as original input by reindexing to original index if original had explicit index
#     out = df_i.reset_index(drop=True)
#     # Recreate the original shape: ensure it has same number of rows as before if original index used; but we often want the scheduled rows kept
#     # We'll return df_i.reset_index(drop=True) but keep Date column (so other code that groups by Date still works)
#     return df_i.reset_index(drop=True).copy()

# def apply_target_average(df, target_avg=None, column=None, lock_schedules=False):
#     """
#     Scale a column in df to match target_avg.
#     If lock_schedules=True, it will try to preserve points where values were explicitly set (i.e., large jumps).
#     Note: target_avg expected to be average over the dataframe (not per-day).
#     """
#     if df is None or df.empty or column not in df.columns or target_avg is None:
#         return df
#     d = df.copy()
#     cur_avg = d[column].mean()
#     if cur_avg <= 0:
#         return d
#     scale = float(target_avg) / float(cur_avg)
#     if not lock_schedules:
#         d[column] = d[column] * scale
#     else:
#         # Attempt to detect scheduled points as places with sudden changes (non-small diffs)
#         diffs = d[column].diff().abs()
#         # threshold: use median*5 or minimal fallback
#         thresh = max(diffs.median() * 5.0, 1e-6)
#         scheduled_idx = diffs[diffs > thresh].index.tolist()
#         # Scale only indices that are not scheduled
#         mask = ~d.index.isin(scheduled_idx)
#         d.loc[mask, column] = d.loc[mask, column] * scale
#     return d

# # ------------------ UI SETUP ------------------
# st.set_page_config(page_title="Production Manager", layout="wide")
# if 'auth' not in st.session_state:
#     st.session_state['auth'] = {'logged_in': False}

# #########################################  LOGIN SECTION  #####################################
# import smtplib
# import hashlib, time, os
# from email.mime.text import MIMEText
# import hashlib, time, os
# from datetime import datetime, timedelta
# from email.mime.multipart import MIMEMultipart
# import streamlit as st





# def send_email(to_email, subject, body):
#     sender_email = "hafsatuxy@gmail.com"   # must be verified in Brevo
#     password = "OKnmRy6V7fUY509I"
#     smtp_server = "smtp-relay.brevo.com"
#     smtp_port = 587

#     try:
#         msg = MIMEMultipart()
#         msg["From"] = sender_email
#         msg["To"] = to_email
#         msg["Subject"] = subject
#         msg.attach(MIMEText(body, "plain"))

#         server = smtplib.SMTP(smtp_server, smtp_port)
#         server.starttls()
#         server.login("96cf99001@smtp-brevo.com", password)  # login always with Brevo login
#         server.sendmail(sender_email, to_email, msg.as_string())
#         server.quit()

#         print("✅ Email sent successfully!")
#         return True
#     except Exception as e:
#         print("❌ Failed to send email:", e)
#         return False

# # ------------------ SIDEBAR: AUTH, DURATION ------------------
# st.sidebar.header("Account")

# # action = st.sidebar.selectbox("Action", ["Login", "Sign up", "Forgot password", "Verify account"])
# action = st.sidebar.selectbox(
#     "Action",
#     ["Login", "Sign up", "Forgot password", "Verify account", "Delete account"]
# )


# # ---------- SIGN UP ----------
# # Ensure user is always defined to avoid NameError
# user = st.session_state.get('auth', {}).get('user', None)





# def create_user(username, email, password, is_admin=0, is_verified=0):
#     password_hash = hash_password(password)
#     try:
#         cur.execute(
#             """
#             INSERT INTO users (username, email, password_hash, is_admin, is_verified, verify_token, reset_token)
#             VALUES (?, ?, ?, ?, ?, NULL, NULL)
#             """,
#             (username, email, password_hash, is_admin, is_verified)
#         )
#         conn.commit()
#         return True, "User created successfully"
#     except Exception as e:
#         return False, str(e)







# if action == 'Sign up':
#     st.sidebar.markdown("Create a new account")
#     su_user = st.sidebar.text_input("Username", key='su_user')
#     su_email = st.sidebar.text_input("Email", key='su_email')
#     su_pass = st.sidebar.text_input("Password", type='password', key='su_pass')
#     su_pass2 = st.sidebar.text_input("Confirm password", type='password', key='su_pass2')
    
#     if st.sidebar.button("Create account", key="create_account_btn"):
#         if su_pass != su_pass2:
#             st.sidebar.error("Passwords do not match")
#         elif not su_user or not su_pass or not su_email:
#             st.sidebar.error("Fill all fields")
#         else:
#             # 🚫 Force all signups as normal users
#             is_admin_flag = 0  

#             ok, msg = create_user(
#                 su_user, su_email, su_pass,
#                 is_admin=is_admin_flag,
#                 is_verified=0
#             )

#             if ok:
#                 # Generate verification token
#                 token = hashlib.sha1((su_email + str(time.time())).encode()).hexdigest()[:8]
#                 cur.execute("UPDATE users SET verify_token=? WHERE email=?", (token, su_email))
#                 conn.commit()

#                 # Send email with token
#                 send_email(
#                     su_email,
#                     "Verify Your Account - Production App",
#                     f"Hello {su_user},\n\nThank you for signing up.\n\n"
#                     f"Your verification code is: {token}\n\n"
#                     f"Enter this code in the app to activate your account."
#                 )
#                 st.success(f"✅ Signup successful! A verification email was sent to {su_email}. Please check your inbox (or spam).")
#             else:
#                 st.sidebar.error(msg)

# # ---------- VERIFY ACCOUNT ----------
# elif action == "Verify account":
#     st.sidebar.markdown("Enter the code sent to your email")
#     ver_email = st.sidebar.text_input("Email", key="ver_email")
#     ver_code = st.sidebar.text_input("Verification code", key="ver_code")

#     if st.sidebar.button("Verify now", key="verify_btn"):
#         cur.execute("SELECT id, verify_token FROM users WHERE email=?", (ver_email,))
#         r = cur.fetchone()
#         if r and r[1] == ver_code:
#             cur.execute("UPDATE users SET is_verified=1, verify_token=NULL WHERE id=?", (r[0],))
#             conn.commit()
#             st.sidebar.success("✅ Account verified. You can now login.")
#         else:
#             st.sidebar.error("❌ Invalid verification code")

#     # 🔹 Send verification code
#     if st.sidebar.button("Send Verification Code", key="resend_btn"):
#         cur.execute("SELECT verify_token FROM users WHERE email=?", (ver_email,))
#         r = cur.fetchone()

#         if r:
#             code = r[0]
#             subject = "Your Verification Code"
#             body = f"Hello,\n\nYour verification code is: {code}\n\nUse this to verify your account."
#             try:
#                 send_email(ver_email, subject, body)
#                 st.sidebar.success("📩 Verification code sent to your email!")
#             except Exception as e:
#                 st.sidebar.error(f"❌ Failed to send code: {e}")
#         else:
#             st.sidebar.warning("⚠️ Email not found. Please sign up first.")

# # ---------- FORGOT PASSWORD ----------
# elif action == 'Forgot password':
#     st.sidebar.markdown("Reset password")
#     fp_email = st.sidebar.text_input("Enter your email", key='fp_email')

#     if st.sidebar.button("Send reset code", key="reset_code_btn"):
#         cur.execute("SELECT id FROM users WHERE email=?", (fp_email,))
#         r = cur.fetchone()
#         if r:
#             token = hashlib.sha1((fp_email + str(time.time())).encode()).hexdigest()[:8]

#             cur.execute("UPDATE users SET reset_token=? WHERE id=?", (token, r[0]))
#             conn.commit()

#             send_email(
#                 fp_email,
#                 "Password Reset - Production App",
#                 f"Your reset code is: {token}\n\n"
#                 "Use this code in the app to set a new password."
#             )
#             st.sidebar.success("A reset code has been sent to your email.")
#         else:
#             st.sidebar.error("Email not found")

#     st.sidebar.markdown("---")
#     token_user = st.sidebar.text_input("User ID (from your account)", key="token_user")
#     token_val = st.sidebar.text_input("Reset code", key="token_val")
#     token_newpw = st.sidebar.text_input("New password", type='password', key="token_newpw")

#     if st.sidebar.button("Reset password now", key="reset_now_btn"):
#         try:
#             uid = int(token_user)
#             cur.execute("SELECT reset_token FROM users WHERE id=?", (uid,))
#             db_token = cur.fetchone()
#             if db_token and db_token[0] == token_val:
#                 ph = hash_password(token_newpw)
#                 cur.execute("UPDATE users SET password_hash=?, reset_token=NULL WHERE id=?", (ph, uid))
#                 conn.commit()
#                 st.sidebar.success("Password updated. Please login.")
#             else:
#                 st.sidebar.error("Invalid or expired token")
#         except Exception:
#             st.sidebar.error("Invalid user id")

# # ---------- LOGIN ----------
# elif action == "Login":
#     login_user = st.sidebar.text_input("Username or email", key='login_user')
#     login_pass = st.sidebar.text_input("Password", type='password', key='login_pass')

#     dur_map = {
#         "1 minute": 60, "5 minutes": 300, "30 minutes": 1800,
#         "1 hour": 3600, "1 day": 86400, "7 days": 604800, "30 days": 2592000
#     }
#     sel_dur = st.sidebar.selectbox("Login duration", list(dur_map.keys()), index=3)

#     if st.sidebar.button("Login", key="login_btn"):
#         ok, user = authenticate(login_user, login_pass)
#         if ok:
#             # ✅ Ensure verified check works properly
#             if not user.get('is_verified') or user['is_verified'] == 0:
#                 st.sidebar.error("⚠️ Please verify your email before logging in.")
#             else:
#                 st.session_state['auth'] = {
#                     'logged_in': True,
#                     'user': user,
#                     'expires_at': (datetime.utcnow() + timedelta(seconds=dur_map[sel_dur])).timestamp()
#                 }
#                 # st.sidebar.success(f"✅ Logged in as {user['username']}")

#                 # Send login notification email
#                 send_email(
#                     user['email'],
#                     "Login Notification - Production App",
#                     f"Hello {user['username']},\n\n"
#                     f"Your account was just logged in on {datetime.utcnow()} (UTC)."
#                 )
#         else:
#             st.sidebar.error("❌ Invalid credentials. If you are a new user, please sign up.")



#     # # --- After login/signup, check session ---
#     # auth = st.session_state.get("auth", {})
#     # user = auth.get("user")

#     # Add these two lines (safe global access to session_state auth)
#     auth = st.session_state.get("auth", {})
#     user = auth.get("user")

#     if not auth.get("logged_in"):
#         # Not logged in yet
#         st.title("Welcome — To the Production App")
#         st.write("Please sign up or login from the sidebar.")
#         st.stop()  # 🔴 Prevents the rest of the app from running
#     else:
#         # ✅ Logged in successfully → safe to use user['id']
#         st.sidebar.success(f"Logged in as {user['username']}")

#         # Example: show uploads for this user
#         cur.execute(
#             "SELECT id, filename, filepath, field_name, uploaded_at FROM uploads WHERE user_id=? ORDER BY uploaded_at DESC",
#             (user['id'],) if user else (-1,)

#         )
#         uploads = cur.fetchall()

#         st.write("### Your Uploads")
#         # if uploads:
#         #     for up in uploads:
#         #         st.write(f"📂 {up[1]} (uploaded {up[4]})")
#         # else:
#         #     st.write("No uploads yet.")
#         if uploads:
#            df = pd.DataFrame(uploads, columns=["ID", "Filename", "Path", "Field", "Uploaded At"])
#            st.dataframe(df)
#         else:
#             st.write("No uploads yet.")



#         if st.sidebar.button("Logout", key="logout_btn"):
#             st.session_state["auth"] = {"logged_in": False, "user": None}
#             st.experimental_rerun()

# elif action == "Delete account":
#     st.sidebar.markdown("Delete your account permanently")

#     del_user = st.sidebar.text_input("Username or Email", key="del_user")
#     del_pass = st.sidebar.text_input("Password", type="password", key="del_pass")

#     if st.sidebar.button("Delete my account", key="del_btn"):
#         ok, user = authenticate(del_user, del_pass)
#         if ok and user:
#             # Delete account from DB
#             cur.execute("DELETE FROM users WHERE id=?", (user['id'],) if user else (-1,)
# )
#             conn.commit()

#             # Clear session
#             if "auth" in st.session_state:
#                 st.session_state.pop("auth")

#             st.sidebar.success("Your account has been permanently deleted.")
#             st.session_state["user"] = None

#             # Optional: send email notification
#             send_email(
#                 user['email'],
#                 "Account Deleted - Production App",
#                 f"Hello {user['username']},\n\nYour account has been permanently deleted."
#             )

#         else:
#             st.sidebar.error("Invalid username/email or password")

# # ------------------ SIDEBAR: Deferments (100) ------------------
# with st.sidebar.expander("⚙️ Deferment Controls", expanded=False):
#     if "deferments" not in st.session_state:
#         st.session_state["deferments"] = {}

#     options = ["None", "Maintenance", "Pipeline Issue", "Reservoir", "Other"]
#     for i in range(1, 6):  # reduced to 5 in UI for practicality; original allowed 100. adjust as needed
#         st.markdown(f"**Deferment {i}**")
#         reason = st.selectbox(f"Reason {i}", options, key=f"deferment_reason_{i}")
#         start_date = st.date_input(f"Start Date {i}", key=f"deferment_start_{i}")
#         duration = st.number_input(f"Duration (days) {i}", min_value=0, max_value=3650, step=1, key=f"deferment_duration_{i}")
#         st.session_state["deferments"][f"Deferment {i}"] = {
#             "reason": reason,
#             "start_date": str(start_date),
#             "duration_days": int(duration)
#         }

#     st.markdown("### 📋 Active Deferments")
#     active = {k:v for k,v in st.session_state["deferments"].items() if v["reason"] != "None" and v["duration_days"] > 0}
#     if active:
#         for name, d in active.items():
#             st.markdown(f"- **{name}** → {d['reason']} starting {d['start_date']} for {d['duration_days']} days")
#     else:
#         st.info("No active deferments set")

# # ===========================
# # SIDEBAR CONTROLS - RAMP & DECLINE
# # ===========================
# with st.sidebar.expander("⚡ Ramp-up & Decline Controls", expanded=False):
#     if "rampups" not in st.session_state:
#         st.session_state["rampups"] = {}
#     if "declines" not in st.session_state:
#         st.session_state["declines"] = {}
#     # input for a single ramp or decline then add to session_state (keeps original UI)
#     ramp_date = st.date_input("Ramp-up Start Date", key="ramp_date")
#     ramp_rate = st.number_input("New Ramp-up Rate (absolute)", min_value=0.0, value=5000.0, key="ramp_rate")
#     if st.button("Apply Ramp-up"):
#         st.session_state["rampups"][f"ramp_{len(st.session_state['rampups'])+1}"] = {
#             "date": str(ramp_date),
#             "rate": float(ramp_rate)
#         }
#         st.success(f"Ramp-up set: {ramp_date} → {ramp_rate}")

#     st.markdown("---")
#     decline_date = st.date_input("Decline Start Date", key="decline_date")
#     decline_pct = st.slider("Decline Percentage (%)", 0, 100, 10, key="decline_pct")
#     if st.button("Apply Decline"):
#         st.session_state["declines"][f"decline_{len(st.session_state['declines'])+1}"] = {
#             "date": str(decline_date),
#             "pct": int(decline_pct)
#         }
#         st.success(f"Decline set: {decline_date} → -{decline_pct}%")

#     st.markdown("---")
#     st.write("Active Ramps")
#     for k, v in st.session_state.get("rampups", {}).items():
#         st.write(f"- {k}: {v.get('date')} → {v.get('rate')}")

#     st.write("Active Declines")
#     for k, v in st.session_state.get("declines", {}).items():
#         st.write(f"- {k}: {v.get('date')} → -{v.get('pct')}%")

#     st.markdown("---")
#     st.write
#     ("Target avg (optional) — applied after scheduling")
#     if "target_avg" not in st.session_state:
#         st.session_state["target_avg"] = {"Oil": None, "Gas": None, "lock": True}
#     ta_oil = st.number_input("Target Avg Oil (BOPD) — leave 0 to ignore", min_value=0, value=int(st.session_state["target_avg"].get("Oil") or 0), key="ta_oil")
#     ta_gas = st.number_input("Target Avg Gas (MMSCFD) — leave 0 to ignore", min_value=0, value=int(st.session_state["target_avg"].get("Gas") or 0), key="ta_gas")
#     lock_sched = st.checkbox("Lock scheduled points (do not rescale them)", value=st.session_state["target_avg"].get("lock", True))
#     if st.button("Set Target Avg"):
#         st.session_state["target_avg"]["Oil"] = int(ta_oil) if ta_oil > 0 else None
#         st.session_state["target_avg"]["Gas"] = int(ta_gas) if ta_gas > 0 else None
#         st.session_state["target_avg"]["lock"] = bool(lock_sched)
#         st.success("Target averages stored in session")




# # --- After ALL auth actions (login/signup/verify/forgot/delete), check login state ---
# auth = st.session_state.get("auth", {})
# user = auth.get("user")

# if not auth.get("logged_in"):
#     # Not logged in yet
#     st.title("Welcome — To the Production App")
#     st.write("Please sign up or login from the sidebar.")
#     st.stop()  # 🔴 Prevents the rest of the app from running
# else:
#     # Logged in successfully → safe to use user['id']
#     # st.sidebar.success(f"Logged in as {user['username']}")

#     # Example: show uploads for this user
#     cur.execute(
#         "SELECT id, filename, filepath, field_name, uploaded_at FROM uploads WHERE user_id=? ORDER BY uploaded_at DESC",
#         (user['id'],) if user else (-1,)

#     )
#     uploads = cur.fetchall()

#     st.write("### Your Uploads")
#     # if uploads:
#     #     for up in uploads:
#     #         st.write(f"📂 {up[1]} (uploaded {up[4]})")
#     # else:
#     #     st.write("No uploads yet.")
#     if uploads:
#         df = pd.DataFrame(uploads, columns=["ID", "Filename", "Path", "Field", "Uploaded At"])
#         st.dataframe(df)
#     else:
#         st.write("No uploads yet.")

# # --- Final global auth check ---
# auth = st.session_state.get("auth", {})
# user = auth.get("user")

# if not auth.get("logged_in"):
#     st.title("Welcome — To the Production App")
#     st.write("Please sign up or login from the sidebar.")
#     st.stop()
# else:
#     st.sidebar.success(f"✅ Logged in as {user['username']}")

#     # Admin-only DB path
#     if user and user.get("is_admin"):
#         st.sidebar.write(f"Local DB: {DB_PATH}")

#     # Show uploads
#     cur.execute(
#         "SELECT id, filename, filepath, field_name, uploaded_at FROM uploads WHERE user_id=? ORDER BY uploaded_at DESC",
#         (user['id'],)
#     )
#     uploads = cur.fetchall()

#     st.write("### Your Uploads")
#     if uploads:
#         df = pd.DataFrame(uploads, columns=["ID","Filename","Path","Field","Uploaded At"])
#         st.dataframe(df)
#     else:
#         st.write("No uploads yet.")

# ########################
# # If Admin, allow them to switch into a user's workspace
# if user and user.get("is_admin"):
#     st.sidebar.markdown("### 👥 Admin Workspace Switch")
#     cur.execute("SELECT id, username FROM users ORDER BY username ASC")
#     all_users = cur.fetchall()
#     user_map = {u[1]: u[0] for u in all_users}

#     selected_username = st.sidebar.selectbox(
#         "Select user workspace to view",
#         ["-- My own (Admin) --"] + list(user_map.keys())
#     )

#     if selected_username != "-- My own (Admin) --":
#         cur.execute("SELECT * FROM users WHERE id=?", (user_map[selected_username],))
#         impersonated_user = cur.fetchone()
#         if impersonated_user:
#             # Override `user` object
#             user = dict(zip([d[0] for d in cur.description], impersonated_user))

#             # 🔔 Banner across the app
#             st.markdown(
#                 f"""
#                 <div style="background-color:#ffecb3;
#                             padding:10px;
#                             border-radius:8px;
#                             border:1px solid #f0ad4e;
#                             margin-bottom:15px;">
#                     ⚠️ <b>Admin Mode:</b> You are impersonating <b>{user['username']}</b>.
#                     All actions (uploads, deletes, deferments) will be performed as this user.
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )


# # ------------------ MAIN TABS ------------------
# st.title("Production App — Workspaces")
# tabs = st.tabs(["Production Input", "Forecast Analysis", "Account & Admin", "Recent Uploads", "Saved Files"])

# # ------------------ PRODUCTION INPUT ------------------
# with tabs[0]:
#     st.header("Production Input")
#     st.write("Upload CSV files or do manual entry. Required columns: Date, Oil (BOPD), Gas (MMSCFD).")

#     # Let user choose method
#     method = st.radio("Choose Input Method", ["Upload CSV", "Manual Entry"], horizontal=True)

#     if method == "Upload CSV":
#         uploaded = st.file_uploader("Upload CSV", type=['csv'], key='upl1')
#         field_name = st.text_input("Field name (e.g. OML 98)")
#         notes = st.text_area("Notes (optional)")
#         if st.button("Upload and validate"):
#             if uploaded is None:
#                 st.error("Please select a file to upload.")
#             else:
#                 try:
#                     df = pd.read_csv(uploaded)
#                 except Exception as e:
#                     st.error(f"Could not read CSV: {e}")
#                     df = None
#                 if df is not None:
#                     ok, out = validate_and_normalize_df(df)
#                     if not ok:
#                         st.error(out)
#                     else:
#                         df_clean = out
#                         ts = int(time.time())
#                         filename = f"{user['username']}_{field_name}_{ts}.csv"
#                         filepath = os.path.join(DATA_DIR, filename)
#                         df_clean.to_csv(filepath, index=False)
#                         cur.execute("INSERT INTO uploads (user_id,filename,filepath,uploaded_at,field_name,notes) VALUES (?,?,?,?,?,?)",
#                                     (user['id'], filename, filepath, datetime.utcnow().isoformat(), field_name, notes))
#                         conn.commit()
#                         st.success("Uploaded and saved")
#                         st.dataframe(df_clean.head())

#     else:  # Manual Entry
#         st.subheader("Manual Excel-like Workspace (Paste or Type)")

#         ws_name = st.text_input("Workspace name", key="manual_ws_name")

#         if "manual_table" not in st.session_state:
#             st.session_state["manual_table"] = pd.DataFrame({
#                 "Date": [""] * 8,
#                 "Oil (BOPD)": [None] * 8,
#                 "Gas (MMSCFD)": [None] * 8
#             })

#         st.info("You can paste large blocks (Date, Oil, Gas). Date format YYYY-MM-DD recommended.")
#         edited_df = st.data_editor(st.session_state["manual_table"], num_rows="dynamic", key="manual_table_editor", use_container_width=True)
#         st.session_state["manual_table"] = edited_df

#         if not edited_df.empty:
#             st.subheader(f"Preview — {ws_name if ws_name else 'Unnamed'}")
#             st.dataframe(edited_df.head(50))

#             totals = pd.DataFrame({
#                 "Date": ["TOTAL"],
#                 "Oil (BOPD)": [pd.to_numeric(edited_df["Oil (BOPD)"], errors="coerce").sum(skipna=True)],
#                 "Gas (MMSCFD)": [pd.to_numeric(edited_df["Gas (MMSCFD)"], errors="coerce").sum(skipna=True)]
#             })
#             st.write("**Totals:**")
#             st.dataframe(totals)

#             if st.button("Save Workspace to CSV"):
#                 if not ws_name.strip():
#                     st.error("Please enter a workspace name.")
#                 else:
#                     ok, norm = validate_and_normalize_df(edited_df)
#                     if not ok:
#                         st.error(norm)
#                     else:
#                         ts = int(time.time())
#                         filename = f"{user['username']}_{ws_name}_{ts}.csv"
#                         filepath = os.path.join(DATA_DIR, filename)
#                         norm.to_csv(filepath, index=False)
#                         cur.execute("INSERT INTO uploads (user_id,filename,filepath,uploaded_at,field_name,notes) VALUES (?,?,?,?,?,?)",
#                                     (user['id'], filename, filepath, datetime.utcnow().isoformat(), ws_name, "manual workspace"))
#                         conn.commit()
#                         st.success(f"Workspace saved as {filename}")
# ############################################################################################################
# # ------------------ FORECAST ANALYSIS ------------------
# with tabs[1]:
#     st.header("Forecast Analysis")
#     st.write("Combine uploaded files (or pick a file) and run analysis. Deferments will zero production in selected windows and be shaded on plots.")

#     # # List user's uploads
#     # cur.execute("SELECT id,filename,filepath,field_name,uploaded_at FROM uploads WHERE user_id=? ORDER BY uploaded_at DESC", (user['id'],))
#     # myfiles = cur.fetchall()
#     # files_df = pd.DataFrame(myfiles, columns=['id','filename','filepath','field_name','uploaded_at']) if myfiles else pd.DataFrame()

#     files_df = pd.DataFrame()  # ✅ always define first

#     if user:  # <-- Only run if logged in
#         # List user's uploads
#         cur.execute("""
#             SELECT id,filename,filepath,field_name,uploaded_at 
#             FROM uploads 
#             WHERE user_id=? 
#             ORDER BY uploaded_at DESC
#         """,(user['id'],))

#         myfiles = cur.fetchall()
#         files_df = pd.DataFrame(myfiles, columns=['id','filename','filepath','field_name','uploaded_at']) if myfiles else pd.DataFrame()

#     if files_df.empty:
#         st.info("No uploads found. Use Production Input to upload or create a manual workspace.")
#     else:
#         sel_files = st.multiselect("Select files to include in analysis (multiple allowed)", files_df['filename'].tolist())
#         if sel_files:
#             dfs = []
#             for fn in sel_files:
#                 fp = files_df.loc[files_df['filename'] == fn, 'filepath'].values[0]
#                 if os.path.exists(fp):
#                     df = pd.read_csv(fp, parse_dates=['Date'])
#                     df['source_file'] = fn
#                     dfs.append(df)
#             if not dfs:
#                 st.error("Selected files not found on disk.")
#             else:
#                 big = pd.concat(dfs, ignore_index=True)
#                 big['Date'] = pd.to_datetime(big['Date'], errors='coerce')
#                 st.subheader("Combined preview")
#                 st.dataframe(big.head(50))

#                 freq = st.selectbox("Aggregation freq", ['D', 'M', 'Y'], index=1, help='D=Daily, M=Monthly, Y=Yearly')
#                 horizon_choice = st.selectbox("Forecast horizon unit", ['Days', 'Months', 'Years'], index=2)
#                 horizon_value = st.number_input("Horizon amount (integer)", min_value=1, max_value=100000, value=10)
#                 if st.button("Run analysis"):
#                     if freq == 'M':
#                         big['period'] = big['Date'].dt.to_period('M').dt.to_timestamp()
#                     elif freq == 'Y':
#                         big['period'] = big['Date'].dt.to_period('A').dt.to_timestamp()
#                     else:
#                         big['period'] = big['Date']

#                     agg = big.groupby('period')[['Oil (BOPD)', 'Gas (MMSCFD)']].sum().reset_index().rename(columns={'period':'Date'})
#                     agg['Date'] = pd.to_datetime(agg['Date'])
#                     st.session_state['agg_cache'] = agg.copy()  # cache for KNN section
#                     st.subheader('Aggregated series')

#                     # Apply deferments to aggregated series for visualization
#                     deferments = st.session_state.get("deferments", {})
#                     agg_adj = apply_deferments(agg, deferments)

#                     # Apply scheduled ramp-ups/declines to aggregated (historical) series for visualization
#                     # Use st.session_state ramp/decline dicts directly; function will interpret the dict structure
#                     agg_adj = apply_scheduled_changes(agg_adj, st.session_state.get("rampups", {}), "Oil (BOPD)")
#                     agg_adj = apply_scheduled_changes(agg_adj, st.session_state.get("declines", {}), "Oil (BOPD)")
#                     agg_adj = apply_scheduled_changes(agg_adj, st.session_state.get("rampups", {}), "Gas (MMSCFD)")
#                     agg_adj = apply_scheduled_changes(agg_adj, st.session_state.get("declines", {}), "Gas (MMSCFD)")

#                     # The target average stored (if any) should be applied optionally for display
#                     ta = st.session_state.get("target_avg", {})
#                     if ta:
#                         if ta.get("Oil"):
#                             agg_adj = apply_target_average(agg_adj, target_avg=ta.get("Oil"), column="Oil (BOPD)", lock_schedules=ta.get("lock", True))
#                         if ta.get("Gas"):
#                             agg_adj = apply_target_average(agg_adj, target_avg=ta.get("Gas"), column="Gas (MMSCFD)", lock_schedules=ta.get("lock", True))

#                     st.line_chart(agg_adj.set_index('Date')[['Oil (BOPD)', 'Gas (MMSCFD)']])

#                     # Matplotlib plot with shading (historical)
#                     fig, ax = plt.subplots(figsize=(10,4))
#                     ax.plot(agg_adj['Date'], agg_adj['Oil (BOPD)'], label='Oil (BOPD)')
#                     ax.plot(agg_adj['Date'], agg_adj['Gas (MMSCFD)'], label='Gas (MMSCFD)')
#                     shade_deferment_spans(ax, deferments)
#                     ax.set_xlabel('Date'); ax.set_ylabel('Production')
#                     ax.legend(); ax.grid(True)
#                     st.pyplot(fig)

#     # --- KNN extended forecasting (re-usable) ---
#     st.markdown("---")
#     st.subheader("KNN-based Extended Forecast")
#     knn_source_df = st.session_state.get("agg_cache", None)
#     if knn_source_df is None and not files_df.empty:
#         pick = st.selectbox("Pick single uploaded file for KNN (if no agg cached)", files_df['filename'].tolist(), key='knn_pick')
#         if pick:
#             fp = files_df.loc[files_df['filename']==pick,'filepath'].values[0]
#             if os.path.exists(fp):
#                 tmp = pd.read_csv(fp, parse_dates=['Date'])
#                 tmp = tmp.groupby('Date')[['Oil (BOPD)','Gas (MMSCFD)']].sum().reset_index()
#                 knn_source_df = tmp.sort_values('Date')
#     if knn_source_df is None or knn_source_df.empty:
#         st.info("No data available for KNN. Create/upload and aggregate first.")
#     else:
#         series_choice = st.radio("Model series", ["Oil (BOPD)", "Gas (MMSCFD)"], horizontal=True)
#         df_prod = knn_source_df[['Date', series_choice]].rename(columns={series_choice:'Production'})

#         # Apply deferments to the series
#         df_prod = apply_deferments(df_prod, st.session_state.get("deferments", {}))

#         if df_prod.empty:
#             st.warning("No data points to model.")
#         else:
#             if not SKLEARN_AVAILABLE:
#                 st.error("scikit-learn not available. `pip install scikit-learn` to enable KNN forecasting.")
#             else:
#                 with st.expander("KNN Settings"):
#                     max_val = int(df_prod['Production'].max() * 2) if df_prod['Production'].max() > 0 else 10000
#                     target_avg = st.slider("Target Average (BOPD/MMscfd)", 0, max_val, min(4500, max_val//2), 100)
#                     n_neighbors = st.slider("KNN neighbors", 1, 20, 3)
#                     extend_years = st.slider("Forecast horizon (years)", 1, 75, 10)

#                 df_prod = df_prod.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
#                 lock_end = df_prod['Date'].max()
#                 hist = df_prod.copy()
#                 hist['Days'] = (hist['Date'] - hist['Date'].min()).dt.days
#                 X = hist[['Days']]
#                 y = hist['Production'].fillna(0)
#                 knn = KNeighborsRegressor(n_neighbors=n_neighbors)
#                 knn.fit(X, y)

#                 future_end = lock_end + pd.DateOffset(years=int(extend_years))
#                 future_days = pd.date_range(start=lock_end + pd.Timedelta(days=1), end=future_end, freq='D')
#                 future_X = (future_days - hist['Date'].min()).days.values.reshape(-1,1)
#                 future_pred = knn.predict(future_X)
#                 future_df = pd.DataFrame({'Date': future_days, 'Production': future_pred})
#                 forecast_df = pd.concat([hist[['Date','Production']], future_df], ignore_index=True).reset_index(drop=True)

#                 # *** APPLY SCHEDULES TO KNN FORECAST & HISTORICAL PRODUCTION ***
#                 # Our schedules in session_state are dicts like {'ramp_1': {'date': 'YYYY-MM-DD', 'rate': 5000}, ...}
#                 # apply_scheduled_changes can accept these dicts and will compute absolute targets or convert pct declines.
#                 # Apply to historical part first (for display / editing)
#                 hist_part = forecast_df[forecast_df['Date'] <= lock_end].reset_index(drop=True)
#                 hist_part = apply_scheduled_changes(hist_part, st.session_state.get("rampups", {}), "Production")
#                 hist_part = apply_scheduled_changes(hist_part, st.session_state.get("declines", {}), "Production")

#                 # Then apply to forecast part
#                 fcst_part = forecast_df[forecast_df['Date'] > lock_end].reset_index(drop=True)
#                 # When schedules reference dates beyond history, apply_scheduled_changes will create/insert rows and interpolate
#                 fcst_part = apply_scheduled_changes(fcst_part, st.session_state.get("rampups", {}), "Production")
#                 fcst_part = apply_scheduled_changes(fcst_part, st.session_state.get("declines", {}), "Production")

#                 # Recombine and ensure ordering
#                 forecast_df = pd.concat([hist_part, fcst_part], ignore_index=True).sort_values('Date').reset_index(drop=True)

#                 # Apply deferments to forecast (set to 0)
#                 deferments = st.session_state.get("deferments", {})
#                 defer_dates = []
#                 for d in deferments.values():
#                     if d.get('reason') and d.get('reason') != 'None' and int(d.get('duration_days',0))>0:
#                         sd = pd.to_datetime(d.get('start_date'), errors='coerce')
#                         if pd.isna(sd):
#                             continue
#                         days = int(d.get('duration_days',0))
#                         defer_dates += pd.date_range(sd, periods=days, freq='D').tolist()
#                 if defer_dates:
#                     forecast_df['Deferred'] = forecast_df['Date'].isin(defer_dates)
#                     forecast_df.loc[forecast_df['Deferred'], 'Production'] = 0
#                 else:
#                     forecast_df['Deferred'] = False

#                 # Rescale future non-deferred days to hit target_avg (over whole forecast_df)
#                 total_days = len(forecast_df)
#                 if total_days>0:
#                     hist_mask = forecast_df['Date'] <= lock_end
#                     hist_cum = forecast_df.loc[hist_mask, 'Production'].sum()
#                     required_total = target_avg * total_days
#                     required_future_prod = required_total - hist_cum
#                     valid_future_mask = (forecast_df['Date'] > lock_end) & (~forecast_df['Deferred'])
#                     num_valid = valid_future_mask.sum()
#                     if num_valid > 0:
#                         new_avg = required_future_prod / num_valid
#                         forecast_df.loc[valid_future_mask, 'Production'] = new_avg

#                 # Now apply global target average if set in sidebar session (this keeps original behavior)
#                 ta = st.session_state.get("target_avg", {})
#                 if ta and ta.get("Oil") and series_choice == "Oil (BOPD)":
#                     forecast_df = apply_target_average(forecast_df, target_avg=ta.get("Oil"), column="Production", lock_schedules=ta.get("lock", True))
#                 if ta and ta.get("Gas") and series_choice == "Gas (MMSCFD)":
#                     forecast_df = apply_target_average(forecast_df, target_avg=ta.get("Gas"), column="Production", lock_schedules=ta.get("lock", True))

#                 forecast_df['Year'] = forecast_df['Date'].dt.year
#                 min_year = int(forecast_df['Year'].min())
#                 max_year = int(forecast_df['Year'].max())
#                 sel_years = st.slider("Select Year Range", min_year, max_year, (min_year, max_year))
#                 analysis_df = forecast_df[(forecast_df['Year'] >= sel_years[0]) & (forecast_df['Year'] <= sel_years[1])]

#                 # metrics & plot
#                 st.metric("Cumulative Production", f"{analysis_df['Production'].sum():,.0f}")
#                 fig, ax = plt.subplots(figsize=(12,4))
#                 ax.plot(analysis_df['Date'], analysis_df['Production'], label='Production', color="green")
#                 ax.axhline(target_avg, linestyle='--', label='Target Avg', color="red")
#                 ax.axvline(lock_end, linestyle='--', label='End of History', color="black")
#                 shade_deferment_spans(ax, deferments)
#                 ax.set_title(f"KNN Forecast — {series_choice}")
#                 ax.set_xlabel("Date"); ax.set_ylabel(series_choice)
#                 ax.legend(); ax.grid(True)
#                 st.pyplot(fig)

#                 # editable tables
#                 st.subheader("Edit Historical")
#                 historical_data = analysis_df[analysis_df['Date'] <= lock_end][['Date','Production','Deferred']]
#                 hist_edit = st.data_editor(historical_data, num_rows='dynamic', key='knn_hist_editor')
#                 st.subheader("Edit Forecast")
#                 forecast_only = analysis_df[analysis_df['Date'] > lock_end][['Date','Production','Deferred']]
#                 forecast_only = forecast_only[~forecast_only['Date'].isin(hist_edit['Date'])]
#                 fcst_edit = st.data_editor(forecast_only, num_rows='dynamic', key='knn_fcst_editor')

#                 merged = pd.concat([hist_edit, fcst_edit], ignore_index=True).sort_values('Date')
#                 st.subheader("Forecast Data (editable)")
#                 st.dataframe(merged, hide_index=True)

#                 # downloads
#                 csv_data = merged.to_csv(index=False).encode('utf-8')
#                 excel_buf = BytesIO()
#                 with pd.ExcelWriter(excel_buf, engine='xlsxwriter') as writer:
#                     merged.to_excel(writer, sheet_name='Forecast', index=False)
#                 st.download_button("Download CSV", data=csv_data, file_name='forecast.csv')
#                 st.download_button("Download Excel", data=excel_buf.getvalue(), file_name='forecast.xlsx')

# # # ------------------ ACCOUNT & ADMIN ------------------
# # with tabs[2]:
# #     st.header("Account & Admin")
# #     st.subheader("Account info")
# #     st.write(user)
# #     st.markdown("---")
# #     # if user['is_admin']:
# #     if user and user.get('is_admin'):

# #         st.subheader("Admin: view all uploads")
# #         cur.execute("SELECT u.id,u.username,a.filename,a.uploaded_at,a.field_name,a.filepath FROM users u JOIN uploads a ON u.id=a.user_id ORDER BY a.uploaded_at DESC LIMIT 500")
# #         allrows = cur.fetchall()
# #         if allrows:
# #             adf = pd.DataFrame(allrows, columns=['user_id','username','filename','uploaded_at','field_name','filepath'])
# #             st.dataframe(adf)
# #             if st.button("Download combined CSV of all uploads"):
# #                 frames = []
# #                 for fp in adf['filepath']:
# #                     if os.path.exists(fp):
# #                         frames.append(pd.read_csv(fp))
# #                 if frames:
# #                     combined = pd.concat(frames, ignore_index=True)
# #                     buf = BytesIO()
# #                     combined.to_csv(buf, index=False)
# #                     buf.seek(0)
# #                     st.download_button("Download combined CSV (final)", data=buf, file_name='combined_all_users.csv')
# #                 else:
# #                     st.warning("No files found on disk")
# #         else:
# #             st.info("No uploads yet")
# #     else:
# #         st.info("You are not an admin. Admins can view/download uploads.")

# # st.sidebar.markdown('---')
# # st.sidebar.write(f"Local DB: {DB_PATH}")

# # import os

# # # Directory where files are stored
# # DATA_DIR = os.path.join(os.getcwd(), "uploaded_files")
# #######################################
# # # ------------------ ACCOUNT & ADMIN ------------------
# # with tabs[2]:
# #     st.header("Account & Admin")

# #     st.subheader("Account info")
# #     if user:
# #         st.json(user)
# #     else:
# #         st.warning("⚠️ No user logged in")
# #     st.markdown("---")

# #     # Admin features
# #     if user and user.get('is_admin'):
# #         st.subheader("Admin: view all uploads")
# #         cur.execute("""
# #             SELECT u.id as user_id, u.username, a.filename, a.uploaded_at, a.field_name, a.filepath
# #             FROM users u 
# #             JOIN uploads a ON u.id = a.user_id 
# #             ORDER BY a.uploaded_at DESC 
# #             LIMIT 500
# #         """)
# #         allrows = cur.fetchall()

# #         if allrows:
# #             adf = pd.DataFrame(allrows, columns=['user_id','username','filename','uploaded_at','field_name','filepath'])
# #             st.dataframe(adf)

# #             if st.button("Download combined CSV of all uploads"):
# #                 frames = []
# #                 for fp in adf['filepath']:
# #                     if os.path.exists(fp):
# #                         frames.append(pd.read_csv(fp))
# #                 if frames:
# #                     combined = pd.concat(frames, ignore_index=True)
# #                     buf = BytesIO()
# #                     combined.to_csv(buf, index=False)
# #                     buf.seek(0)
# #                     st.download_button(
# #                         "Download combined CSV (final)",
# #                         data=buf,
# #                         file_name='combined_all_users.csv'
# #                     )
# #                 else:
# #                     st.warning("No files found on disk")
# #         else:
# #             st.info("No uploads yet")
# #     else:
# #         st.info("You are not an admin. Admins can view/download uploads.")

# # st.sidebar.markdown('---')
# # if user and user.get("is_admin"):
# #     st.sidebar.write(f"Local DB: {DB_PATH}")

# # # Directory where files are stored
# # DATA_DIR = os.path.join(os.getcwd(), "uploaded_files")
# #######################################
# import logging

# # ------------------ LOGGING SETUP ------------------
# LOG_FILE = "app_errors.log"
# logging.basicConfig(
#     filename=LOG_FILE,
#     level=logging.ERROR,
#     format="%(asctime)s [%(levelname)s] %(message)s"
# )

# # ------------------ ACCOUNT & ADMIN ------------------
# with tabs[2]:
#     st.header("Account & Admin")

#     st.subheader("Account info")
#     if user:
#         st.json(user)
#     else:
#         st.warning("⚠️ No user logged in")
#     st.markdown("---")

#     # Admin features
#     if user and user.get('is_admin'):
#         st.subheader("🔑 Admin: Manage All Uploads")

#         try:
#             # Get all users
#             cur.execute("SELECT id, username FROM users ORDER BY username ASC")
#             all_users = cur.fetchall()
#         except Exception as e:
#             logging.error(f"Failed to fetch users: {e}", exc_info=True)
#             all_users = []

#         if all_users:
#             # User filter dropdown
#             usernames = {u[1]: u[0] for u in all_users}
#             selected_user = st.selectbox("👤 Select a user", list(usernames.keys()))

#             # Fetch that user’s uploads
#             cur.execute(
#                 "SELECT filename, uploaded_at, field_name, filepath FROM uploads WHERE user_id=? ORDER BY uploaded_at DESC",
#                 (usernames[selected_user],)
#             )
#             user_uploads = cur.fetchall()

#             if user_uploads:
#                 df_user = pd.DataFrame(user_uploads, columns=['Filename', 'Uploaded At', 'Field', 'Filepath'])
#                 st.dataframe(df_user[['Filename', 'Uploaded At', 'Field']])

#                 # Pick a file
#                 file_choice = st.selectbox("📂 Select a file to view", df_user['Filename'])

#                 # Show preview or allow download
#                 chosen_row = df_user[df_user['Filename'] == file_choice].iloc[0]
#                 filepath = chosen_row['Filepath']

#                 if os.path.exists(filepath):
#                     try:
#                         df_preview = pd.read_csv(filepath)
#                         st.write("📊 Preview of selected file:")
#                         st.dataframe(df_preview.head(20))  # show first 20 rows

#                         # Allow download
#                         with open(filepath, "rb") as f:
#                             st.download_button(
#                                 label="⬇️ Download file",
#                                 data=f,
#                                 file_name=chosen_row['Filename'],
#                                 mime="text/csv"
#                             )
#                     except Exception as e:
#                         logging.error(f"Failed to read file {filepath}: {e}", exc_info=True)
#                         st.error("⚠️ Could not open this file.")
#                 else:
#                     st.error("⚠️ File not found on disk.")
#             else:
#                 st.info("This user has no uploads.")
#         else:
#             st.info("No users found in system.")



# #################
# # # ------------------ ACCOUNT & ADMIN ------------------
# # with tabs[2]:
# #     st.header("Account & Admin")

# #     st.subheader("Account info")
# #     if user:
# #         st.json(user)
# #     else:
# #         st.warning("⚠️ No user logged in")
# #     st.markdown("---")

# #     # Admin features
# #     if user and user.get('is_admin'):
# #         st.subheader("Admin: view all uploads")

# #         try:
# #             cur.execute("""
# #                 SELECT u.id as user_id, u.username, a.filename, a.uploaded_at, a.field_name, a.filepath
# #                 FROM users u 
# #                 JOIN uploads a ON u.id = a.user_id 
# #                 ORDER BY a.uploaded_at DESC 
# #                 LIMIT 500
# #             """)
# #             allrows = cur.fetchall()
# #         except Exception as e:
# #             logging.error(f"Database query failed: {e}", exc_info=True)
# #             st.error("⚠️ Could not load uploads. Please try again later.")
# #             allrows = []

# #         if allrows:
# #             adf = pd.DataFrame(allrows, columns=['user_id','username','filename','uploaded_at','field_name','filepath'])
# #             st.dataframe(adf)

# #             if st.button("Download combined CSV of all uploads"):
# #                 frames = []
# #                 for fp in adf['filepath']:
# #                     if os.path.exists(fp):
# #                         try:
# #                             frames.append(pd.read_csv(fp))
# #                         except Exception as e:
# #                             logging.error(f"Failed to read file {fp}: {e}", exc_info=True)
# #                             st.warning(f"⚠️ Skipped a corrupted/missing file.")
# #                 if frames:
# #                     combined = pd.concat(frames, ignore_index=True)
# #                     buf = BytesIO()
# #                     combined.to_csv(buf, index=False)
# #                     buf.seek(0)
# #                     st.download_button(
# #                         "Download combined CSV (final)",
# #                         data=buf,
# #                         file_name='combined_all_users.csv'
# #                     )
# #                 else:
# #                     st.warning("No files found on disk")
# #         else:
# #             st.info("No uploads yet")
# #     else:
# #         st.info("You are not an admin. Admins can view/download uploads.")

# # st.sidebar.markdown('---')
# # if user and user.get("is_admin"):
# #     st.sidebar.write(f"Local DB: {DB_PATH}")

# # # Directory where files are stored
# # DATA_DIR = os.path.join(os.getcwd(), "uploaded_files")




# # # ------------------ RECENT UPLOADS TAB ------------------
# # with tabs[3]:
# #     st.header("🕒 Recent Uploads")

# #     if user["is_admin"]:
# #         cur.execute("""
# #             SELECT u.filename, u.field_name, u.notes, u.uploaded_at, usr.username
# #             FROM uploads u
# #             JOIN users usr ON u.user_id = usr.id
# #             ORDER BY u.uploaded_at DESC
# #             LIMIT 10
# #         """)
# #         rows = cur.fetchall()

# #         if rows:
# #             st.subheader("All Users' Recent Uploads (Admin)")
# #             for idx, r in enumerate(rows):
# #                 col1, col2, col3, col4 = st.columns([2,3,2,2])
# #                 col1.write(r[0])   # filename
# #                 col2.write(r[4])   # uploaded_by
# #                 col3.write(r[3])   # uploaded_at

# #                 # Use separate keys
# #                 confirm_state_key = f"admin_confirm_state_{idx}"
# #                 delete_button_key = f"admin_delete_btn_{idx}"
# #                 confirm_button_key = f"admin_confirm_btn_{idx}"

# #                 if st.session_state.get(confirm_state_key, False):
# #                     if col4.button("❌ Confirm Delete", key=confirm_button_key):
# #                         # Delete from DB
# #                         cur.execute("DELETE FROM uploads WHERE filename=?", (r[0],))
# #                         conn.commit()

# #                         # Delete from disk
# #                         file_path = os.path.join(DATA_DIR, r[0])
# #                         if os.path.exists(file_path):
# #                             os.remove(file_path)

# #                         st.warning(f"Admin deleted {r[0]}")
# #                         st.session_state[confirm_state_key] = False
# #                         st.experimental_rerun()
# #                 else:
# #                     if col4.button("🗑️ Delete", key=delete_button_key):
# #                         st.session_state[confirm_state_key] = True
# #                         st.error(f"⚠️ Click confirm to delete {r[0]}")
# #         else:
# #             st.info("No recent uploads in the system.")

# #     else:
# #         cur.execute(
# #             "SELECT filename, field_name, notes, uploaded_at FROM uploads WHERE user_id = ? ORDER BY uploaded_at DESC LIMIT 5", 
# #             (user['id'],) if user else (-1,)

# #         )
# #         rows = cur.fetchall()

# #         if rows:
# #             st.subheader("My Recent Uploads")
# #             for idx, r in enumerate(rows):
# #                 col1, col2, col3 = st.columns([3,3,2])
# #                 col1.write(r[0])   # filename
# #                 col2.write(r[3])   # uploaded_at

# #                 # Use separate keys
# #                 confirm_state_key = f"user_confirm_state_{idx}"
# #                 delete_button_key = f"user_delete_btn_{idx}"
# #                 confirm_button_key = f"user_confirm_btn_{idx}"

# #                 if st.session_state.get(confirm_state_key, False):
# #                     if col3.button("❌ Confirm Delete", key=confirm_button_key):
# #                         # Delete from DB
# #                         cur.execute("DELETE FROM uploads WHERE filename=? AND user_id=?", (r[0], user['id']))
# #                         conn.commit()

# #                         # Delete from disk
# #                         file_path = os.path.join(DATA_DIR, r[0])
# #                         if os.path.exists(file_path):
# #                             os.remove(file_path)

# #                         st.success(f"Deleted {r[0]}")
# #                         st.session_state[confirm_state_key] = False
# #                         st.experimental_rerun()
# #                 else:
# #                     if col3.button("🗑️ Delete", key=delete_button_key):
# #                         st.session_state[confirm_state_key] = True
# #                         st.error(f"⚠️ Click confirm to delete {r[0]}")
# #         else:
# #             st.info("You have no recent uploads.")
# #################################################################
# # ------------------ RECENT UPLOADS TAB ------------------
# with tabs[3]:
#     st.header("🕒 Recent Uploads")

#     if user:  # ✅ only run if someone is logged in
#         if user.get("is_admin"):
#             cur.execute("""
#                 SELECT u.filename, u.field_name, u.notes, u.uploaded_at, usr.username
#                 FROM uploads u
#                 JOIN users usr ON u.user_id = usr.id
#                 ORDER BY u.uploaded_at DESC
#                 LIMIT 10
#             """)
#             rows = cur.fetchall()

#             if rows:
#                 st.subheader("All Users' Recent Uploads (Admin)")
#                 for idx, r in enumerate(rows):
#                     col1, col2, col3, col4 = st.columns([2,3,2,2])
#                     col1.write(r[0])   # filename
#                     col2.write(r[4])   # uploaded_by
#                     col3.write(r[3])   # uploaded_at

#                     # Use separate keys
#                     confirm_state_key = f"admin_confirm_state_{idx}"
#                     delete_button_key = f"admin_delete_btn_{idx}"
#                     confirm_button_key = f"admin_confirm_btn_{idx}"

#                     if st.session_state.get(confirm_state_key, False):
#                         if col4.button("❌ Confirm Delete", key=confirm_button_key):
#                             # Delete from DB
#                             cur.execute("DELETE FROM uploads WHERE filename=?", (r[0],))
#                             conn.commit()

#                             # Delete from disk
#                             file_path = os.path.join(DATA_DIR, r[0])
#                             if os.path.exists(file_path):
#                                 os.remove(file_path)

#                             st.warning(f"Admin deleted {r[0]}")
#                             st.session_state[confirm_state_key] = False
#                             st.experimental_rerun()
#                     else:
#                         if col4.button("🗑️ Delete", key=delete_button_key):
#                             st.session_state[confirm_state_key] = True
#                             st.error(f"⚠️ Click confirm to delete {r[0]}")
#             else:
#                 st.info("No recent uploads in the system.")

#         else:  # normal user
#             cur.execute(
#                 "SELECT filename, field_name, notes, uploaded_at FROM uploads WHERE user_id = ? ORDER BY uploaded_at DESC LIMIT 5", 
#                 (user['id'],)
#             )
#             rows = cur.fetchall()

#             if rows:
#                 st.subheader("My Recent Uploads")
#                 for idx, r in enumerate(rows):
#                     col1, col2, col3 = st.columns([3,3,2])
#                     col1.write(r[0])   # filename
#                     col2.write(r[3])   # uploaded_at

#                     # Use separate keys
#                     confirm_state_key = f"user_confirm_state_{idx}"
#                     delete_button_key = f"user_delete_btn_{idx}"
#                     confirm_button_key = f"user_confirm_btn_{idx}"

#                     if st.session_state.get(confirm_state_key, False):
#                         if col3.button("❌ Confirm Delete", key=confirm_button_key):
#                             # Delete from DB
#                             cur.execute("DELETE FROM uploads WHERE filename=? AND user_id=?", (r[0], user['id']))
#                             conn.commit()

#                             # Delete from disk
#                             file_path = os.path.join(DATA_DIR, r[0])
#                             if os.path.exists(file_path):
#                                 os.remove(file_path)

#                             st.success(f"Deleted {r[0]}")
#                             st.session_state[confirm_state_key] = False
#                             st.experimental_rerun()
#                     else:
#                         if col3.button("🗑️ Delete", key=delete_button_key):
#                             st.session_state[confirm_state_key] = True
#                             st.error(f"⚠️ Click confirm to delete {r[0]}")
#             else:
#                 st.info("You have no recent uploads.")
#     else:
#         st.info("Please log in to view recent uploads.")  # ✅ safe when not logged in





# # # ------------------ SAVED FILES TAB ------------------
# # with tabs[4]:  # adjust index depending on your layout query language
# #     st.header("📂 Saved Files")

# #     if user["is_admin"]:
# #         cur.execute("""
# #             SELECT u.filepath, u.filename, u.field_name, u.notes, u.uploaded_at, usr.username
# #             FROM uploads u
# #             JOIN users usr ON u.user_id = usr.id
# #             ORDER BY u.uploaded_at DESC
# #         """)
# #         rows = cur.fetchall()

# #         if rows:
# #             df_files = pd.DataFrame(rows, columns=["Filepath", "Filename", "Field", "Notes", "Uploaded At", "Uploaded By"])
# #             for idx, row in df_files.iterrows():
# #                 st.markdown(f"**{row['Filename']}** — Field: {row['Field']} | By: {row['Uploaded By']} | {row['Uploaded At']}")
# #                 try:
# #                     with open(row["Filepath"], "rb") as f:
# #                         st.download_button(
# #                             label="⬇️ Download",
# #                             data=f,
# #                             file_name=row["Filename"],
# #                             mime="text/csv",
# #                             key=f"download_{idx}"
# #                         )
# #                 except FileNotFoundError:
# #                     st.error(f"File {row['Filename']} not found on disk.")
# #         else:
# #             st.info("No files saved in the system yet.")
# #     else:
# #         cur.execute(
# #             "SELECT filepath, filename, field_name, notes, uploaded_at FROM uploads WHERE user_id = ? ORDER BY uploaded_at DESC",
# #             (user['id'],) if user else (-1,)

# #         )
# #         rows = cur.fetchall()

# #         if rows:
# #             df_files = pd.DataFrame(rows, columns=["Filepath", "Filename", "Field", "Notes", "Uploaded At"])
# #             for idx, row in df_files.iterrows():
# #                 st.markdown(f"**{row['Filename']}** — Field: {row['Field']} | {row['Uploaded At']}")
# #                 try:
# #                     with open(row["Filepath"], "rb") as f:
# #                         st.download_button(
# #                             label="⬇️ Download",
# #                             data=f,
# #                             file_name=row["Filename"],
# #                             mime="text/csv",
# #                             key=f"user_download_{idx}"
# #                         )
# #                 except FileNotFoundError:
# #                     st.error(f"File {row['Filename']} not found on disk.")
# #         else:
# #             st.info("You have not saved any files yet.")
# #############################
# # ------------------ SAVED FILES TAB ------------------
# with tabs[4]:  # adjust index depending on your layout
#     st.header("📂 Saved Files")

#     if user:  # ✅ only run if logged in
#         if user.get("is_admin"):
#             cur.execute("""
#                 SELECT u.filepath, u.filename, u.field_name, u.notes, u.uploaded_at, usr.username
#                 FROM uploads u
#                 JOIN users usr ON u.user_id = usr.id
#                 ORDER BY u.uploaded_at DESC
#             """)
#             rows = cur.fetchall()

#             if rows:
#                 df_files = pd.DataFrame(
#                     rows, 
#                     columns=["Filepath", "Filename", "Field", "Notes", "Uploaded At", "Uploaded By"]
#                 )
#                 for idx, row in df_files.iterrows():
#                     st.markdown(f"**{row['Filename']}** — Field: {row['Field']} | By: {row['Uploaded By']} | {row['Uploaded At']}")
#                     try:
#                         with open(row["Filepath"], "rb") as f:
#                             st.download_button(
#                                 label="⬇️ Download",
#                                 data=f,
#                                 file_name=row["Filename"],
#                                 mime="text/csv",
#                                 key=f"download_{idx}"
#                             )
#                     except FileNotFoundError:
#                         st.error(f"File {row['Filename']} not found on disk.")
#             else:
#                 st.info("No files saved in the system yet.")
#         else:  # normal user
#             cur.execute(
#                 "SELECT filepath, filename, field_name, notes, uploaded_at FROM uploads WHERE user_id = ? ORDER BY uploaded_at DESC",
#                 (user['id'],)
#             )
#             rows = cur.fetchall()

#             if rows:
#                 df_files = pd.DataFrame(
#                     rows, 
#                     columns=["Filepath", "Filename", "Field", "Notes", "Uploaded At"]
#                 )
#                 for idx, row in df_files.iterrows():
#                     st.markdown(f"**{row['Filename']}** — Field: {row['Field']} | {row['Uploaded At']}")
#                     try:
#                         with open(row["Filepath"], "rb") as f:
#                             st.download_button(
#                                 label="⬇️ Download",
#                                 data=f,
#                                 file_name=row["Filename"],
#                                 mime="text/csv",
#                                 key=f"user_download_{idx}"
#                             )
#                     except FileNotFoundError:
#                         st.error(f"File {row['Filename']} not found on disk.")
#             else:
#                 st.info("You have not saved any files yet.")
#     else:
#         st.info("Please log in to view saved files.")  # ✅ safe when not logged in




###############################################################################################################################
#########################################################################################################
######################################################################
#####################################################################
################################################
# production_manager_full.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import hashlib
import time
from datetime import datetime, timedelta
from io import BytesIO
import matplotlib.pyplot as plt


# optional: KNN
try:
    from sklearn.neighbors import KNeighborsRegressor
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ------------------ CONFIG ------------------
DB_PATH = "production_app.db"
DATA_DIR = "data_uploads"
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------ DB ------------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()


# # ------------------ DB ------------------
# conn = sqlite3.connect(DB_PATH, check_same_thread=False)
# cur = conn.cursor()


# import sqlite3

# DB_PATH = "production_app.db"

# def reset_users_table():
#     # Use a separate connection just for this operation
#     with sqlite3.connect(DB_PATH) as conn:
#         cur = conn.cursor()

#         # Drop the table if it exists
#         cur.execute("DROP TABLE IF EXISTS users")

#         # Recreate the table
#         cur.execute('''
#         CREATE TABLE users (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             username TEXT UNIQUE,
#             email TEXT UNIQUE,
#             password_hash TEXT,
#             is_admin INTEGER DEFAULT 0,
#             is_verified INTEGER DEFAULT 0,
#             verify_token TEXT,
#             reset_token TEXT,
#             created_at TEXT
#         )
#         ''')

#         conn.commit()
#         print("✅ Users table has been reset successfully!")

# # Call this function once when you want to reset
# reset_users_table()

# --- Create users table if it doesn't exist ---
cur.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    email TEXT UNIQUE,
    password_hash TEXT,
    is_admin INTEGER DEFAULT 0,
    created_at TEXT
)
''')

# --- Create uploads table if it doesn't exist ---
cur.execute('''
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    filename TEXT,
    filepath TEXT,
    uploaded_at TEXT,
    field_name TEXT,
    notes TEXT,
    source TEXT DEFAULT 'upload',
    FOREIGN KEY(user_id) REFERENCES users(id)
)
''')

conn.commit()

# --- Ensure the users table has the additional columns ---
def ensure_user_columns():
    cur.execute("PRAGMA table_info(users)")
    existing = [row[1] for row in cur.fetchall()]

    if "is_verified" not in existing:
        cur.execute("ALTER TABLE users ADD COLUMN is_verified INTEGER DEFAULT 0")
    if "verify_token" not in existing:
        cur.execute("ALTER TABLE users ADD COLUMN verify_token TEXT")
    if "reset_token" not in existing:
        cur.execute("ALTER TABLE users ADD COLUMN reset_token TEXT")
    conn.commit()

ensure_user_columns()

# ------------------ HELPERS & NEW SCHEDULING LOGIC ------------------
REQUIRED_COLS = ["Date", "Oil (BOPD)", "Gas (MMSCFD)"]

def hash_password(password: str) -> str:
    salt = "static_salt_change_me"  # replace in production
    return hashlib.sha256((salt + password).encode()).hexdigest()

def create_user(username, email, password, is_admin=0):
    ph = hash_password(password)
    try:
        cur.execute("INSERT INTO users (username,email,password_hash,is_admin,created_at) VALUES (?,?,?,?,?)",
                    (username, email, ph, int(is_admin), datetime.utcnow().isoformat()))
        conn.commit()
        return True, "user created"
    except sqlite3.IntegrityError as e:
        return False, str(e)


def authenticate(username_or_email, password):
    ph = hash_password(password)
    cur.execute(
        "SELECT id, username, email, is_admin, is_verified FROM users WHERE (username=? OR email=?) AND password_hash=?",
        (username_or_email, username_or_email, ph)
    )
    row = cur.fetchone()


    cur.execute("SELECT id, username, email, is_verified FROM users")
    print(cur.fetchall())

    if row:
        user = {
            "id": row[0],
            "username": row[1],
            "email": row[2],
            "is_admin": row[3],
            "is_verified": row[4]  # ✅ include this!
        }
        return True, user
    else:
        return False, None

def validate_and_normalize_df(df: pd.DataFrame):
    # Normalize column names and check required
    colmap = {}
    for c in df.columns:
        c_clean = c.strip()
        if c_clean.lower() in [x.lower() for x in REQUIRED_COLS]:
            for req in REQUIRED_COLS:
                if c_clean.lower() == req.lower():
                    colmap[c] = req
        elif 'date' in c_clean.lower():
            colmap[c] = 'Date'
        elif 'oil' in c_clean.lower():
            colmap[c] = 'Oil (BOPD)'
        elif 'gas' in c_clean.lower():
            colmap[c] = 'Gas (MMSCFD)'
        else:
            colmap[c] = c_clean
    df = df.rename(columns=colmap)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    except Exception as e:
        return False, f"Date parse error: {e}"
    if df['Date'].isna().any():
        return False, "Some Date values could not be parsed. Ensure ISO-like dates (YYYY-MM-DD)."
    for c in ['Oil (BOPD)','Gas (MMSCFD)']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        if df[c].isna().all():
            return False, f"Column {c} has no numeric values"
    return True, df

def save_dataframe(user_id, df: pd.DataFrame, field_name: str, notes: str = "", source: str = "upload"):
    ts = int(time.time())
    safe_field = field_name.replace(" ", "_")[:60] if field_name else "unnamed"
    filename = f"user{user_id}_{safe_field}_{ts}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    cur.execute("INSERT INTO uploads (user_id,filename,filepath,uploaded_at,field_name,notes,source) VALUES (?,?,?,?,?,?,?)",
                (user_id, filename, filepath, datetime.utcnow().isoformat(), field_name, notes, source))
    conn.commit()
    return filename, filepath

def apply_deferments(df, deferments):
    """Apply deferments by zeroing Oil & Gas in deferment ranges."""
    if df is None or df.empty or not deferments:
        return df
    df = df.copy()
    if 'Date' not in df.columns:
        return df
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    for d in deferments.values():
        reason = d.get('reason')
        days = int(d.get('duration_days', 0) or 0)
        start_str = d.get('start_date')
        if reason and reason != "None" and days > 0 and start_str:
            try:
                start = pd.to_datetime(start_str, errors='coerce')
                if pd.isna(start):
                    continue
                end = start + pd.Timedelta(days=days-1)
                mask = (df['Date'] >= start) & (df['Date'] <= end)
                if 'Oil (BOPD)' in df.columns:
                    df.loc[mask, 'Oil (BOPD)'] = 0
                if 'Gas (MMSCFD)' in df.columns:
                    df.loc[mask, 'Gas (MMSCFD)'] = 0
                if 'Production' in df.columns:
                    df.loc[mask, 'Production'] = 0
            except Exception:
                continue
    return df

def shade_deferment_spans(ax, deferments, color_map=None):
    spans = []
    for d in deferments.values():
        reason = d.get('reason')
        days = int(d.get('duration_days', 0) or 0)
        start_str = d.get('start_date')
        if reason and reason != "None" and days > 0 and start_str:
            s = pd.to_datetime(start_str, errors='coerce')
            if pd.isna(s):
                continue
            e = s + pd.Timedelta(days=days-1)
            spans.append((s, e, reason))
    for s, e, r in spans:
        color = None
        if color_map and r in color_map:
            color = color_map[r]
        ax.axvspan(s, e, alpha=0.18, color=color)

# ------------------ NEW: SCHEDULING & TARGET AVERAGE HELPERS ------------------

def _ensure_date_index(df):
    """Return a copy indexed by Date (datetime), preserving original Date column too."""
    d = df.copy()
    if 'Date' in d.columns:
        d['Date'] = pd.to_datetime(d['Date'], errors='coerce')
        d = d.sort_values('Date').reset_index(drop=True)
        d_indexed = d.set_index('Date', drop=False)
    else:
        # Try to use existing index
        idx = pd.to_datetime(d.index)
        d_indexed = d.copy()
        d_indexed['Date'] = idx
        d_indexed.index = idx
    return d_indexed

def apply_scheduled_changes(df, schedules, column):
    """
    Apply scheduled ramp-ups/declines with linear interpolation on the specified column.
    schedules: dict where values are dicts (as stored in session_state) OR a mapping date->target
    Accepts input df that has a 'Date' column or datetime index. Returns df with same shape.
    """
    if df is None or df.empty or not schedules or column not in df.columns:
        return df

    # Convert to DataFrame indexed by Date
    df_i = _ensure_date_index(df)

    # Accept two possible schedule formats:
    # 1) state dict as in session_state: {'ramp_1': {'date':'YYYY-MM-DD','rate':5000}, ...}
    # 2) direct mapping: {'2025-09-01': 5000, ...}
    mapping = {}
    # if schedules is a mapping of date->value (string/number)
    if all((isinstance(k, (str, pd.Timestamp)) and not isinstance(v, dict)) for k, v in schedules.items()):
        # treat as direct mapping
        for k, v in schedules.items():
            try:
                d = pd.to_datetime(k)
                mapping[d] = float(v)
            except Exception:
                continue
    else:
        # treat as state dict
        for entry in schedules.values():
            try:
                d = pd.to_datetime(entry.get('date'))
            except Exception:
                continue
            if pd.isna(d):
                continue
            # If entry has 'rate' it's absolute target
            if 'rate' in entry:
                try:
                    mapping[d] = float(entry.get('rate'))
                except Exception:
                    continue
            # If entry has 'pct' treat as a percent change to the base value at that date
            elif 'pct' in entry:
                try:
                    pct = float(entry.get('pct')) / 100.0
                except Exception:
                    pct = 0.0
                # find base value just before or at that date
                if d in df_i.index:
                    base = df_i.loc[d, column]
                    if pd.isna(base):
                        # take previous non-na
                        prev = df_i.loc[:d, column].ffill().iloc[-1] if not df_i.loc[:d, column].ffill().empty else None
                        base = prev if prev is not None else 0.0
                else:
                    # take last known before date
                    prev_series = df_i.loc[df_i.index <= d, column]
                    if not prev_series.empty:
                        base = prev_series.ffill().iloc[-1]
                    else:
                        # fallback: use earliest value
                        base = df_i[column].ffill().iloc[0] if not df_i[column].ffill().empty else 0.0
                mapping[d] = float(base * (1 - pct))

    if not mapping:
        return df

    # Keep only mapping dates within the df index range (we can still insert future dates if df extends into forecast)
    # Insert scheduled targets (create rows if necessary)
    for d, target in mapping.items():
        if d not in df_i.index:
            # if date outside range, add a new row so interpolation can happen
            # create a new row using NaN for all columns, set Date index to d
            new_row = pd.DataFrame([{c: np.nan for c in df_i.columns}])
            new_row.index = pd.Index([d])
            new_row['Date'] = d
            df_i = pd.concat([df_i, new_row])
    # Re-sort index
    df_i = df_i.sort_index()

    # Set scheduled absolute values
    for d, target in mapping.items():
        df_i.at[d, column] = target

    # Interpolate the column across the entire index (linear)
    df_i[column] = df_i[column].interpolate(method='time')  # time-aware interpolation

    # Return df_i with original order and with Date column preserved
    # Remove any rows we artificially added that were outside original df's full date coverage only if original had no such dates
    # To be safe, convert back to same index shape as original input by reindexing to original index if original had explicit index
    out = df_i.reset_index(drop=True)
    # Recreate the original shape: ensure it has same number of rows as before if original index used; but we often want the scheduled rows kept
    # We'll return df_i.reset_index(drop=True) but keep Date column (so other code that groups by Date still works)
    return df_i.reset_index(drop=True).copy()

def apply_target_average(df, target_avg=None, column=None, lock_schedules=False):
    """
    Scale a column in df to match target_avg.
    If lock_schedules=True, it will try to preserve points where values were explicitly set (i.e., large jumps).
    Note: target_avg expected to be average over the dataframe (not per-day).
    """
    if df is None or df.empty or column not in df.columns or target_avg is None:
        return df
    d = df.copy()
    cur_avg = d[column].mean()
    if cur_avg <= 0:
        return d
    scale = float(target_avg) / float(cur_avg)
    if not lock_schedules:
        d[column] = d[column] * scale
    else:
        # Attempt to detect scheduled points as places with sudden changes (non-small diffs)
        diffs = d[column].diff().abs()
        # threshold: use median*5 or minimal fallback
        thresh = max(diffs.median() * 5.0, 1e-6)
        scheduled_idx = diffs[diffs > thresh].index.tolist()
        # Scale only indices that are not scheduled
        mask = ~d.index.isin(scheduled_idx)
        d.loc[mask, column] = d.loc[mask, column] * scale
    return d

# ------------------ UI SETUP ------------------
st.set_page_config(page_title="Production Manager", layout="wide")
if 'auth' not in st.session_state:
    st.session_state['auth'] = {'logged_in': False}

#########################################  LOGIN SECTION  #####################################
import smtplib
import hashlib, time, os
from email.mime.text import MIMEText
import hashlib, time, os
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
import streamlit as st





def send_email(to_email, subject, body):
    sender_email = "hafsatuxy@gmail.com"   # must be verified in Brevo
    password = "OKnmRy6V7fUY509I"
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
        server.login("96cf99001@smtp-brevo.com", password)  # login always with Brevo login
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()

        print("✅ Email sent successfully!")
        return True
    except Exception as e:
        print("❌ Failed to send email:", e)
        return False

# ------------------ SIDEBAR: AUTH, DURATION ------------------
st.sidebar.header("Account")

# action = st.sidebar.selectbox("Action", ["Login", "Sign up", "Forgot password", "Verify account"])
action = st.sidebar.selectbox(
    "Action",
    ["Login", "Sign up", "Forgot password", "Verify account", "Delete account"]
)


# ---------- SIGN UP ----------
# Ensure user is always defined to avoid NameError
user = st.session_state.get('auth', {}).get('user', None)





def create_user(username, email, password, is_admin=0, is_verified=0):
    password_hash = hash_password(password)
    try:
        cur.execute(
            """
            INSERT INTO users (username, email, password_hash, is_admin, is_verified, verify_token, reset_token)
            VALUES (?, ?, ?, ?, ?, NULL, NULL)
            """,
            (username, email, password_hash, is_admin, is_verified)
        )
        conn.commit()
        return True, "User created successfully"
    except Exception as e:
        return False, str(e)







if action == 'Sign up':
    st.sidebar.markdown("Create a new account")
    su_user = st.sidebar.text_input("Username", key='su_user')
    su_email = st.sidebar.text_input("Email", key='su_email')
    su_pass = st.sidebar.text_input("Password", type='password', key='su_pass')
    su_pass2 = st.sidebar.text_input("Confirm password", type='password', key='su_pass2')
    
    if st.sidebar.button("Create account", key="create_account_btn"):
        if su_pass != su_pass2:
            st.sidebar.error("Passwords do not match")
        elif not su_user or not su_pass or not su_email:
            st.sidebar.error("Fill all fields")
        else:
            # 🚫 Force all signups as normal users
            is_admin_flag = 0  

            ok, msg = create_user(
                su_user, su_email, su_pass,
                is_admin=is_admin_flag,
                is_verified=0
            )

            if ok:
                # Generate verification token
                token = hashlib.sha1((su_email + str(time.time())).encode()).hexdigest()[:8]
                cur.execute("UPDATE users SET verify_token=? WHERE email=?", (token, su_email))
                conn.commit()

                # Send email with token
                send_email(
                    su_email,
                    "Verify Your Account - Production App",
                    f"Hello {su_user},\n\nThank you for signing up.\n\n"
                    f"Your verification code is: {token}\n\n"
                    f"Enter this code in the app to activate your account."
                )
                st.success(f"✅ Signup successful! A verification email was sent to {su_email}. Please check your inbox (or spam).")
            else:
                st.sidebar.error(msg)

# ---------- VERIFY ACCOUNT ----------
elif action == "Verify account":
    st.sidebar.markdown("Enter the code sent to your email")
    ver_email = st.sidebar.text_input("Email", key="ver_email")
    ver_code = st.sidebar.text_input("Verification code", key="ver_code")

    if st.sidebar.button("Verify now", key="verify_btn"):
        cur.execute("SELECT id, verify_token FROM users WHERE email=?", (ver_email,))
        r = cur.fetchone()
        if r and r[1] == ver_code:
            cur.execute("UPDATE users SET is_verified=1, verify_token=NULL WHERE id=?", (r[0],))
            conn.commit()
            st.sidebar.success("✅ Account verified. You can now login.")
        else:
            st.sidebar.error("❌ Invalid verification code")

    # 🔹 Send verification code
    if st.sidebar.button("Send Verification Code", key="resend_btn"):
        cur.execute("SELECT verify_token FROM users WHERE email=?", (ver_email,))
        r = cur.fetchone()

        if r:
            code = r[0]
            subject = "Your Verification Code"
            body = f"Hello,\n\nYour verification code is: {code}\n\nUse this to verify your account."
            try:
                send_email(ver_email, subject, body)
                st.sidebar.success("📩 Verification code sent to your email!")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to send code: {e}")
        else:
            st.sidebar.warning("⚠️ Email not found. Please sign up first.")

# ---------- FORGOT PASSWORD ----------
elif action == 'Forgot password':
    st.sidebar.markdown("Reset password")
    fp_email = st.sidebar.text_input("Enter your email", key='fp_email')

    if st.sidebar.button("Send reset code", key="reset_code_btn"):
        cur.execute("SELECT id FROM users WHERE email=?", (fp_email,))
        r = cur.fetchone()
        if r:
            token = hashlib.sha1((fp_email + str(time.time())).encode()).hexdigest()[:8]

            cur.execute("UPDATE users SET reset_token=? WHERE id=?", (token, r[0]))
            conn.commit()

            send_email(
                fp_email,
                "Password Reset - Production App",
                f"Your reset code is: {token}\n\n"
                "Use this code in the app to set a new password."
            )
            st.sidebar.success("A reset code has been sent to your email.")
        else:
            st.sidebar.error("Email not found")

    st.sidebar.markdown("---")
    token_user = st.sidebar.text_input("User ID (from your account)", key="token_user")
    token_val = st.sidebar.text_input("Reset code", key="token_val")
    token_newpw = st.sidebar.text_input("New password", type='password', key="token_newpw")

    if st.sidebar.button("Reset password now", key="reset_now_btn"):
        try:
            uid = int(token_user)
            cur.execute("SELECT reset_token FROM users WHERE id=?", (uid,))
            db_token = cur.fetchone()
            if db_token and db_token[0] == token_val:
                ph = hash_password(token_newpw)
                cur.execute("UPDATE users SET password_hash=?, reset_token=NULL WHERE id=?", (ph, uid))
                conn.commit()
                st.sidebar.success("Password updated. Please login.")
            else:
                st.sidebar.error("Invalid or expired token")
        except Exception:
            st.sidebar.error("Invalid user id")

# ---------- LOGIN ----------
elif action == "Login":
    login_user = st.sidebar.text_input("Username or email", key='login_user')
    login_pass = st.sidebar.text_input("Password", type='password', key='login_pass')

    dur_map = {
        "1 minute": 60, "5 minutes": 300, "30 minutes": 1800,
        "1 hour": 3600, "1 day": 86400, "7 days": 604800, "30 days": 2592000
    }
    sel_dur = st.sidebar.selectbox("Login duration", list(dur_map.keys()), index=3)

    if st.sidebar.button("Login", key="login_btn"):
        ok, user = authenticate(login_user, login_pass)
        if ok:
            # ✅ Ensure verified check works properly
            if not user.get('is_verified') or user['is_verified'] == 0:
                st.sidebar.error("⚠️ Please verify your email before logging in.")
            else:
                st.session_state['auth'] = {
                    'logged_in': True,
                    'user': user,
                    'expires_at': (datetime.utcnow() + timedelta(seconds=dur_map[sel_dur])).timestamp()
                }
                # st.sidebar.success(f"✅ Logged in as {user['username']}")

                # Send login notification email
                send_email(
                    user['email'],
                    "Login Notification - Production App",
                    f"Hello {user['username']},\n\n"
                    f"Your account was just logged in on {datetime.utcnow()} (UTC)."
                )
        else:
            st.sidebar.error("❌ Invalid credentials. If you are a new user, please sign up.")



    # # --- After login/signup, check session ---
    # auth = st.session_state.get("auth", {})
    # user = auth.get("user")

    # Add these two lines (safe global access to session_state auth)
    auth = st.session_state.get("auth", {})
    user = auth.get("user")

    if not auth.get("logged_in"):
        # Not logged in yet
        st.title("Welcome — To the Production App")
        st.write("Please sign up or login from the sidebar.")
        st.stop()  # 🔴 Prevents the rest of the app from running
    else:
        # ✅ Logged in successfully → safe to use user['id']
        st.sidebar.success(f"Logged in as {user['username']}")

        # Example: show uploads for this user
        cur.execute(
            "SELECT id, filename, filepath, field_name, uploaded_at FROM uploads WHERE user_id=? ORDER BY uploaded_at DESC",
            (user['id'],) if user else (-1,)

        )
        uploads = cur.fetchall()

        st.write("### Your Uploads")
        # if uploads:
        #     for up in uploads:
        #         st.write(f"📂 {up[1]} (uploaded {up[4]})")
        # else:
        #     st.write("No uploads yet.")
        if uploads:
           df = pd.DataFrame(uploads, columns=["ID", "Filename", "Path", "Field", "Uploaded At"])
           st.dataframe(df)
        else:
            st.write("No uploads yet.")



        if st.sidebar.button("Logout", key="logout_btn"):
            st.session_state["auth"] = {"logged_in": False, "user": None}
            st.experimental_rerun()

elif action == "Delete account":
    st.sidebar.markdown("Delete your account permanently")

    del_user = st.sidebar.text_input("Username or Email", key="del_user")
    del_pass = st.sidebar.text_input("Password", type="password", key="del_pass")

    if st.sidebar.button("Delete my account", key="del_btn"):
        ok, user = authenticate(del_user, del_pass)
        if ok and user:
            # Delete account from DB
            cur.execute("DELETE FROM users WHERE id=?", (user['id'],) if user else (-1,)
)
            conn.commit()

            # Clear session
            if "auth" in st.session_state:
                st.session_state.pop("auth")

            st.sidebar.success("Your account has been permanently deleted.")
            st.session_state["user"] = None

            # Optional: send email notification
            send_email(
                user['email'],
                "Account Deleted - Production App",
                f"Hello {user['username']},\n\nYour account has been permanently deleted."
            )

        else:
            st.sidebar.error("Invalid username/email or password")

# ------------------ SIDEBAR: Deferments (100) ------------------
with st.sidebar.expander("⚙️ Deferment Controls", expanded=False):
    if "deferments" not in st.session_state:
        st.session_state["deferments"] = {}

    options = ["None", "Maintenance", "Pipeline Issue", "Reservoir", "Other"]
    for i in range(1, 6):  # reduced to 5 in UI for practicality; original allowed 100. adjust as needed
        st.markdown(f"**Deferment {i}**")
        reason = st.selectbox(f"Reason {i}", options, key=f"deferment_reason_{i}")
        start_date = st.date_input(f"Start Date {i}", key=f"deferment_start_{i}")
        duration = st.number_input(f"Duration (days) {i}", min_value=0, max_value=3650, step=1, key=f"deferment_duration_{i}")
        st.session_state["deferments"][f"Deferment {i}"] = {
            "reason": reason,
            "start_date": str(start_date),
            "duration_days": int(duration)
        }

    st.markdown("### 📋 Active Deferments")
    active = {k:v for k,v in st.session_state["deferments"].items() if v["reason"] != "None" and v["duration_days"] > 0}
    if active:
        for name, d in active.items():
            st.markdown(f"- **{name}** → {d['reason']} starting {d['start_date']} for {d['duration_days']} days")
    else:
        st.info("No active deferments set")

# ===========================
# SIDEBAR CONTROLS - RAMP & DECLINE
# ===========================
with st.sidebar.expander("⚡ Ramp-up & Decline Controls", expanded=False):
    if "rampups" not in st.session_state:
        st.session_state["rampups"] = {}
    if "declines" not in st.session_state:
        st.session_state["declines"] = {}
    # input for a single ramp or decline then add to session_state (keeps original UI)
    ramp_date = st.date_input("Ramp-up Start Date", key="ramp_date")
    ramp_rate = st.number_input("New Ramp-up Rate (absolute)", min_value=0.0, value=5000.0, key="ramp_rate")
    if st.button("Apply Ramp-up"):
        st.session_state["rampups"][f"ramp_{len(st.session_state['rampups'])+1}"] = {
            "date": str(ramp_date),
            "rate": float(ramp_rate)
        }
        st.success(f"Ramp-up set: {ramp_date} → {ramp_rate}")

    st.markdown("---")
    decline_date = st.date_input("Decline Start Date", key="decline_date")
    decline_pct = st.slider("Decline Percentage (%)", 0, 100, 10, key="decline_pct")
    if st.button("Apply Decline"):
        st.session_state["declines"][f"decline_{len(st.session_state['declines'])+1}"] = {
            "date": str(decline_date),
            "pct": int(decline_pct)
        }
        st.success(f"Decline set: {decline_date} → -{decline_pct}%")

    st.markdown("---")
    st.write("Active Ramps")
    for k, v in st.session_state.get("rampups", {}).items():
        st.write(f"- {k}: {v.get('date')} → {v.get('rate')}")

    st.write("Active Declines")
    for k, v in st.session_state.get("declines", {}).items():
        st.write(f"- {k}: {v.get('date')} → -{v.get('pct')}%")

    st.markdown("---")
    st.write
    ("Target avg (optional) — applied after scheduling")
    if "target_avg" not in st.session_state:
        st.session_state["target_avg"] = {"Oil": None, "Gas": None, "lock": True}
    ta_oil = st.number_input("Target Avg Oil (BOPD) — leave 0 to ignore", min_value=0, value=int(st.session_state["target_avg"].get("Oil") or 0), key="ta_oil")
    ta_gas = st.number_input("Target Avg Gas (MMSCFD) — leave 0 to ignore", min_value=0, value=int(st.session_state["target_avg"].get("Gas") or 0), key="ta_gas")
    lock_sched = st.checkbox("Lock scheduled points (do not rescale them)", value=st.session_state["target_avg"].get("lock", True))
    if st.button("Set Target Avg"):
        st.session_state["target_avg"]["Oil"] = int(ta_oil) if ta_oil > 0 else None
        st.session_state["target_avg"]["Gas"] = int(ta_gas) if ta_gas > 0 else None
        st.session_state["target_avg"]["lock"] = bool(lock_sched)
        st.success("Target averages stored in session")




# --- After ALL auth actions (login/signup/verify/forgot/delete), check login state ---
auth = st.session_state.get("auth", {})
user = auth.get("user")

if not auth.get("logged_in"):
    # Not logged in yet
    st.title("Welcome — To the Production App")
    st.write("Please sign up or login from the sidebar.")
    st.stop()  # 🔴 Prevents the rest of the app from running
else:
    # Logged in successfully → safe to use user['id']
    # st.sidebar.success(f"Logged in as {user['username']}")

    # Example: show uploads for this user
    cur.execute(
        "SELECT id, filename, filepath, field_name, uploaded_at FROM uploads WHERE user_id=? ORDER BY uploaded_at DESC",
        (user['id'],) if user else (-1,)

    )
    uploads = cur.fetchall()

    st.write("### Your Uploads")
    # if uploads:
    #     for up in uploads:
    #         st.write(f"📂 {up[1]} (uploaded {up[4]})")
    # else:
    #     st.write("No uploads yet.")
    if uploads:
        df = pd.DataFrame(uploads, columns=["ID", "Filename", "Path", "Field", "Uploaded At"])
        st.dataframe(df)
    else:
        st.write("No uploads yet.")

# --- Final global auth check ---
auth = st.session_state.get("auth", {})
user = auth.get("user")

if not auth.get("logged_in"):
    st.title("Welcome — To the Production App")
    st.write("Please sign up or login from the sidebar.")
    st.stop()
else:
    st.sidebar.success(f"✅ Logged in as {user['username']}")

    # Admin-only DB path
    if user and user.get("is_admin"):
        st.sidebar.write(f"Local DB: {DB_PATH}")

    # Show uploads
    cur.execute(
        "SELECT id, filename, filepath, field_name, uploaded_at FROM uploads WHERE user_id=? ORDER BY uploaded_at DESC",
        (user['id'],)
    )
    uploads = cur.fetchall()

    st.write("### Your Uploads")
    if uploads:
        df = pd.DataFrame(uploads, columns=["ID","Filename","Path","Field","Uploaded At"])
        st.dataframe(df)
    else:
        st.write("No uploads yet.")

########################
# If Admin, allow them to switch into a user's workspace
if user and user.get("is_admin"):
    st._main.markdown("### 👥 Admin Workspace Switch")
    cur.execute("SELECT id, username FROM users ORDER BY username ASC")
    all_users = cur.fetchall()
    user_map = {u[1]: u[0] for u in all_users}

    selected_username = st._main.selectbox(
        "Select user workspace to view",
        ["-- My own (Admin) --"] + list(user_map.keys())
    )

    if selected_username != "-- My own (Admin) --":
        cur.execute("SELECT * FROM users WHERE id=?", (user_map[selected_username],))
        impersonated_user = cur.fetchone()
        if impersonated_user:
            # Override `user` object
            user = dict(zip([d[0] for d in cur.description], impersonated_user))

            # 🔔 Banner across the app
            st.markdown(
                f"""
                <div style="background-color:#ffecb3;
                            padding:10px;
                            border-radius:8px;
                            border:1px solid #f0ad4e;
                            margin-bottom:15px;">
                    ⚠️ <b>Admin Mode:</b> You are impersonating <b>{user['username']}</b>.
                    All actions (uploads, deletes, deferments) will be performed as this user.
                </div>
                """,
                unsafe_allow_html=True
            )


# ------------------ MAIN TABS ------------------
st.title("Production App — Workspaces")
tabs = st.tabs(["Production Input", "Forecast Analysis", "Account & Admin", "Recent Uploads", "Saved Files"])

# ------------------ PRODUCTION INPUT ------------------
with tabs[0]:
    st.header("Production Input")
    st.write("Upload CSV files or do manual entry. Required columns: Date, Oil (BOPD), Gas (MMSCFD).")

    # Let user choose method
    method = st.radio("Choose Input Method", ["Upload CSV", "Manual Entry"], horizontal=True)

    if method == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=['csv'], key='upl1')
        field_name = st.text_input("Field name (e.g. OML 98)")
        notes = st.text_area("Notes (optional)")
        if st.button("Upload and validate"):
            if uploaded is None:
                st.error("Please select a file to upload.")
            else:
                try:
                    df = pd.read_csv(uploaded)
                except Exception as e:
                    st.error(f"Could not read CSV: {e}")
                    df = None
                if df is not None:
                    ok, out = validate_and_normalize_df(df)
                    if not ok:
                        st.error(out)
                    else:
                        df_clean = out
                        ts = int(time.time())
                        filename = f"{user['username']}_{field_name}_{ts}.csv"
                        filepath = os.path.join(DATA_DIR, filename)
                        df_clean.to_csv(filepath, index=False)
                        cur.execute("INSERT INTO uploads (user_id,filename,filepath,uploaded_at,field_name,notes) VALUES (?,?,?,?,?,?)",
                                    (user['id'], filename, filepath, datetime.utcnow().isoformat(), field_name, notes))
                        conn.commit()
                        st.success("Uploaded and saved")
                        st.dataframe(df_clean.head())

    else:  # Manual Entry
        st.subheader("Manual Excel-like Workspace (Paste or Type)")

        ws_name = st.text_input("Workspace name", key="manual_ws_name")

        if "manual_table" not in st.session_state:
            st.session_state["manual_table"] = pd.DataFrame({
                "Date": [""] * 8,
                "Oil (BOPD)": [None] * 8,
                "Gas (MMSCFD)": [None] * 8
            })

        st.info("You can paste large blocks (Date, Oil, Gas). Date format YYYY-MM-DD recommended.")
        edited_df = st.data_editor(st.session_state["manual_table"], num_rows="dynamic", key="manual_table_editor", use_container_width=True)
        st.session_state["manual_table"] = edited_df

        if not edited_df.empty:
            st.subheader(f"Preview — {ws_name if ws_name else 'Unnamed'}")
            st.dataframe(edited_df.head(50))

            totals = pd.DataFrame({
                "Date": ["TOTAL"],
                "Oil (BOPD)": [pd.to_numeric(edited_df["Oil (BOPD)"], errors="coerce").sum(skipna=True)],
                "Gas (MMSCFD)": [pd.to_numeric(edited_df["Gas (MMSCFD)"], errors="coerce").sum(skipna=True)]
            })
            st.write("**Totals:**")
            st.dataframe(totals)

            if st.button("Save Workspace to CSV"):
                if not ws_name.strip():
                    st.error("Please enter a workspace name.")
                else:
                    ok, norm = validate_and_normalize_df(edited_df)
                    if not ok:
                        st.error(norm)
                    else:
                        ts = int(time.time())
                        filename = f"{user['username']}_{ws_name}_{ts}.csv"
                        filepath = os.path.join(DATA_DIR, filename)
                        norm.to_csv(filepath, index=False)
                        cur.execute("INSERT INTO uploads (user_id,filename,filepath,uploaded_at,field_name,notes) VALUES (?,?,?,?,?,?)",
                                    (user['id'], filename, filepath, datetime.utcnow().isoformat(), ws_name, "manual workspace"))
                        conn.commit()
                        st.success(f"Workspace saved as {filename}")
############################################################################################################
# ------------------ FORECAST ANALYSIS ------------------
with tabs[1]:
    st.header("Forecast Analysis")
    st.write("Combine uploaded files (or pick a file) and run analysis. Deferments will zero production in selected windows and be shaded on plots.")

    # # List user's uploads
    # cur.execute("SELECT id,filename,filepath,field_name,uploaded_at FROM uploads WHERE user_id=? ORDER BY uploaded_at DESC", (user['id'],))
    # myfiles = cur.fetchall()
    # files_df = pd.DataFrame(myfiles, columns=['id','filename','filepath','field_name','uploaded_at']) if myfiles else pd.DataFrame()

    files_df = pd.DataFrame()  # ✅ always define first

    if user:  # <-- Only run if logged in
        # List user's uploads
        cur.execute("""
            SELECT id,filename,filepath,field_name,uploaded_at 
            FROM uploads 
            WHERE user_id=? 
            ORDER BY uploaded_at DESC
        """,(user['id'],))

        myfiles = cur.fetchall()
        files_df = pd.DataFrame(myfiles, columns=['id','filename','filepath','field_name','uploaded_at']) if myfiles else pd.DataFrame()

    if files_df.empty:
        st.info("No uploads found. Use Production Input to upload or create a manual workspace.")
    else:
        sel_files = st.multiselect("Select files to include in analysis (multiple allowed)", files_df['filename'].tolist())
        if sel_files:
            dfs = []
            for fn in sel_files:
                fp = files_df.loc[files_df['filename'] == fn, 'filepath'].values[0]
                if os.path.exists(fp):
                    df = pd.read_csv(fp, parse_dates=['Date'])
                    df['source_file'] = fn
                    dfs.append(df)
            if not dfs:
                st.error("Selected files not found on disk.")
            else:
                big = pd.concat(dfs, ignore_index=True)
                big['Date'] = pd.to_datetime(big['Date'], errors='coerce')
                st.subheader("Combined preview")
                st.dataframe(big.head(50))

                freq = st.selectbox("Aggregation freq", ['D', 'M', 'Y'], index=1, help='D=Daily, M=Monthly, Y=Yearly')
                horizon_choice = st.selectbox("Forecast horizon unit", ['Days', 'Months', 'Years'], index=2)
                horizon_value = st.number_input("Horizon amount (integer)", min_value=1, max_value=100000, value=10)
                if st.button("Run analysis"):
                    if freq == 'M':
                        big['period'] = big['Date'].dt.to_period('M').dt.to_timestamp()
                    elif freq == 'Y':
                        big['period'] = big['Date'].dt.to_period('A').dt.to_timestamp()
                    else:
                        big['period'] = big['Date']

                    agg = big.groupby('period')[['Oil (BOPD)', 'Gas (MMSCFD)']].sum().reset_index().rename(columns={'period':'Date'})
                    agg['Date'] = pd.to_datetime(agg['Date'])
                    st.session_state['agg_cache'] = agg.copy()  # cache for KNN section
                    st.subheader('Aggregated series')

                    # Apply deferments to aggregated series for visualization
                    deferments = st.session_state.get("deferments", {})
                    agg_adj = apply_deferments(agg, deferments)

                    # Apply scheduled ramp-ups/declines to aggregated (historical) series for visualization
                    # Use st.session_state ramp/decline dicts directly; function will interpret the dict structure
                    agg_adj = apply_scheduled_changes(agg_adj, st.session_state.get("rampups", {}), "Oil (BOPD)")
                    agg_adj = apply_scheduled_changes(agg_adj, st.session_state.get("declines", {}), "Oil (BOPD)")
                    agg_adj = apply_scheduled_changes(agg_adj, st.session_state.get("rampups", {}), "Gas (MMSCFD)")
                    agg_adj = apply_scheduled_changes(agg_adj, st.session_state.get("declines", {}), "Gas (MMSCFD)")

                    # The target average stored (if any) should be applied optionally for display
                    ta = st.session_state.get("target_avg", {})
                    if ta:
                        if ta.get("Oil"):
                            agg_adj = apply_target_average(agg_adj, target_avg=ta.get("Oil"), column="Oil (BOPD)", lock_schedules=ta.get("lock", True))
                        if ta.get("Gas"):
                            agg_adj = apply_target_average(agg_adj, target_avg=ta.get("Gas"), column="Gas (MMSCFD)", lock_schedules=ta.get("lock", True))

                    st.line_chart(agg_adj.set_index('Date')[['Oil (BOPD)', 'Gas (MMSCFD)']])

                    # Matplotlib plot with shading (historical)
                    fig, ax = plt.subplots(figsize=(10,4))
                    ax.plot(agg_adj['Date'], agg_adj['Oil (BOPD)'], label='Oil (BOPD)')
                    ax.plot(agg_adj['Date'], agg_adj['Gas (MMSCFD)'], label='Gas (MMSCFD)')
                    shade_deferment_spans(ax, deferments)
                    ax.set_xlabel('Date'); ax.set_ylabel('Production')
                    ax.legend(); ax.grid(True)
                    st.pyplot(fig)

    # --- KNN extended forecasting (re-usable) ---
    st.markdown("---")
    st.subheader("KNN-based Extended Forecast")
    knn_source_df = st.session_state.get("agg_cache", None)
    if knn_source_df is None and not files_df.empty:
        pick = st.selectbox("Pick single uploaded file for KNN (if no agg cached)", files_df['filename'].tolist(), key='knn_pick')
        if pick:
            fp = files_df.loc[files_df['filename']==pick,'filepath'].values[0]
            if os.path.exists(fp):
                tmp = pd.read_csv(fp, parse_dates=['Date'])
                tmp = tmp.groupby('Date')[['Oil (BOPD)','Gas (MMSCFD)']].sum().reset_index()
                knn_source_df = tmp.sort_values('Date')
    if knn_source_df is None or knn_source_df.empty:
        st.info("No data available for KNN. Create/upload and aggregate first.")
    else:
        series_choice = st.radio("Model series", ["Oil (BOPD)", "Gas (MMSCFD)"], horizontal=True)
        df_prod = knn_source_df[['Date', series_choice]].rename(columns={series_choice:'Production'})

        # Apply deferments to the series
        df_prod = apply_deferments(df_prod, st.session_state.get("deferments", {}))

        if df_prod.empty:
            st.warning("No data points to model.")
        else:
            if not SKLEARN_AVAILABLE:
                st.error("scikit-learn not available. `pip install scikit-learn` to enable KNN forecasting.")
            else:
                with st.expander("KNN Settings"):
                    max_val = int(df_prod['Production'].max() * 2) if df_prod['Production'].max() > 0 else 10000
                    target_avg = st.slider("Target Average (BOPD/MMscfd)", 0, max_val, min(4500, max_val//2), 100)
                    n_neighbors = st.slider("KNN neighbors", 1, 20, 3)
                    extend_years = st.slider("Forecast horizon (years)", 1, 75, 10)

                df_prod = df_prod.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
                lock_end = df_prod['Date'].max()
                hist = df_prod.copy()
                hist['Days'] = (hist['Date'] - hist['Date'].min()).dt.days
                X = hist[['Days']]
                y = hist['Production'].fillna(0)
                knn = KNeighborsRegressor(n_neighbors=n_neighbors)
                knn.fit(X, y)

                future_end = lock_end + pd.DateOffset(years=int(extend_years))
                future_days = pd.date_range(start=lock_end + pd.Timedelta(days=1), end=future_end, freq='D')
                future_X = (future_days - hist['Date'].min()).days.values.reshape(-1,1)
                future_pred = knn.predict(future_X)
                future_df = pd.DataFrame({'Date': future_days, 'Production': future_pred})
                forecast_df = pd.concat([hist[['Date','Production']], future_df], ignore_index=True).reset_index(drop=True)

                # *** APPLY SCHEDULES TO KNN FORECAST & HISTORICAL PRODUCTION ***
                # Our schedules in session_state are dicts like {'ramp_1': {'date': 'YYYY-MM-DD', 'rate': 5000}, ...}
                # apply_scheduled_changes can accept these dicts and will compute absolute targets or convert pct declines.
                # Apply to historical part first (for display / editing)
                hist_part = forecast_df[forecast_df['Date'] <= lock_end].reset_index(drop=True)
                hist_part = apply_scheduled_changes(hist_part, st.session_state.get("rampups", {}), "Production")
                hist_part = apply_scheduled_changes(hist_part, st.session_state.get("declines", {}), "Production")

                # Then apply to forecast part
                fcst_part = forecast_df[forecast_df['Date'] > lock_end].reset_index(drop=True)
                # When schedules reference dates beyond history, apply_scheduled_changes will create/insert rows and interpolate
                fcst_part = apply_scheduled_changes(fcst_part, st.session_state.get("rampups", {}), "Production")
                fcst_part = apply_scheduled_changes(fcst_part, st.session_state.get("declines", {}), "Production")

                # Recombine and ensure ordering
                forecast_df = pd.concat([hist_part, fcst_part], ignore_index=True).sort_values('Date').reset_index(drop=True)

                # Apply deferments to forecast (set to 0)
                deferments = st.session_state.get("deferments", {})
                defer_dates = []
                for d in deferments.values():
                    if d.get('reason') and d.get('reason') != 'None' and int(d.get('duration_days',0))>0:
                        sd = pd.to_datetime(d.get('start_date'), errors='coerce')
                        if pd.isna(sd):
                            continue
                        days = int(d.get('duration_days',0))
                        defer_dates += pd.date_range(sd, periods=days, freq='D').tolist()
                if defer_dates:
                    forecast_df['Deferred'] = forecast_df['Date'].isin(defer_dates)
                    forecast_df.loc[forecast_df['Deferred'], 'Production'] = 0
                else:
                    forecast_df['Deferred'] = False

                # Rescale future non-deferred days to hit target_avg (over whole forecast_df)
                total_days = len(forecast_df)
                if total_days>0:
                    hist_mask = forecast_df['Date'] <= lock_end
                    hist_cum = forecast_df.loc[hist_mask, 'Production'].sum()
                    required_total = target_avg * total_days
                    required_future_prod = required_total - hist_cum
                    valid_future_mask = (forecast_df['Date'] > lock_end) & (~forecast_df['Deferred'])
                    num_valid = valid_future_mask.sum()
                    if num_valid > 0:
                        new_avg = required_future_prod / num_valid
                        forecast_df.loc[valid_future_mask, 'Production'] = new_avg

                # Now apply global target average if set in sidebar session (this keeps original behavior)
                ta = st.session_state.get("target_avg", {})
                if ta and ta.get("Oil") and series_choice == "Oil (BOPD)":
                    forecast_df = apply_target_average(forecast_df, target_avg=ta.get("Oil"), column="Production", lock_schedules=ta.get("lock", True))
                if ta and ta.get("Gas") and series_choice == "Gas (MMSCFD)":
                    forecast_df = apply_target_average(forecast_df, target_avg=ta.get("Gas"), column="Production", lock_schedules=ta.get("lock", True))

                forecast_df['Year'] = forecast_df['Date'].dt.year
                min_year = int(forecast_df['Year'].min())
                max_year = int(forecast_df['Year'].max())
                sel_years = st.slider("Select Year Range", min_year, max_year, (min_year, max_year))
                analysis_df = forecast_df[(forecast_df['Year'] >= sel_years[0]) & (forecast_df['Year'] <= sel_years[1])]

                # metrics & plot
                st.metric("Cumulative Production", f"{analysis_df['Production'].sum():,.0f}")
                fig, ax = plt.subplots(figsize=(12,4))
                ax.plot(analysis_df['Date'], analysis_df['Production'], label='Production', color="green")
                ax.axhline(target_avg, linestyle='--', label='Target Avg', color="red")
                ax.axvline(lock_end, linestyle='--', label='End of History', color="black")
                shade_deferment_spans(ax, deferments)
                ax.set_title(f"KNN Forecast — {series_choice}")
                ax.set_xlabel("Date"); ax.set_ylabel(series_choice)
                ax.legend(); ax.grid(True)
                st.pyplot(fig)

                # editable tables
                st.subheader("Edit Historical")
                historical_data = analysis_df[analysis_df['Date'] <= lock_end][['Date','Production','Deferred']]
                hist_edit = st.data_editor(historical_data, num_rows='dynamic', key='knn_hist_editor')
                st.subheader("Edit Forecast")
                forecast_only = analysis_df[analysis_df['Date'] > lock_end][['Date','Production','Deferred']]
                forecast_only = forecast_only[~forecast_only['Date'].isin(hist_edit['Date'])]
                fcst_edit = st.data_editor(forecast_only, num_rows='dynamic', key='knn_fcst_editor')

                merged = pd.concat([hist_edit, fcst_edit], ignore_index=True).sort_values('Date')
                st.subheader("Forecast Data (editable)")
                st.dataframe(merged, hide_index=True)

                # downloads
                csv_data = merged.to_csv(index=False).encode('utf-8')
                excel_buf = BytesIO()
                with pd.ExcelWriter(excel_buf, engine='xlsxwriter') as writer:
                    merged.to_excel(writer, sheet_name='Forecast', index=False)
                st.download_button("Download CSV", data=csv_data, file_name='forecast.csv')
                st.download_button("Download Excel", data=excel_buf.getvalue(), file_name='forecast.xlsx')

# # ------------------ ACCOUNT & ADMIN ------------------
# with tabs[2]:
#     st.header("Account & Admin")
#     st.subheader("Account info")
#     st.write(user)
#     st.markdown("---")
#     # if user['is_admin']:
#     if user and user.get('is_admin'):

#         st.subheader("Admin: view all uploads")
#         cur.execute("SELECT u.id,u.username,a.filename,a.uploaded_at,a.field_name,a.filepath FROM users u JOIN uploads a ON u.id=a.user_id ORDER BY a.uploaded_at DESC LIMIT 500")
#         allrows = cur.fetchall()
#         if allrows:
#             adf = pd.DataFrame(allrows, columns=['user_id','username','filename','uploaded_at','field_name','filepath'])
#             st.dataframe(adf)
#             if st.button("Download combined CSV of all uploads"):
#                 frames = []
#                 for fp in adf['filepath']:
#                     if os.path.exists(fp):
#                         frames.append(pd.read_csv(fp))
#                 if frames:
#                     combined = pd.concat(frames, ignore_index=True)
#                     buf = BytesIO()
#                     combined.to_csv(buf, index=False)
#                     buf.seek(0)
#                     st.download_button("Download combined CSV (final)", data=buf, file_name='combined_all_users.csv')
#                 else:
#                     st.warning("No files found on disk")
#         else:
#             st.info("No uploads yet")
#     else:
#         st.info("You are not an admin. Admins can view/download uploads.")

# st.sidebar.markdown('---')
# st.sidebar.write(f"Local DB: {DB_PATH}")

# import os

# # Directory where files are stored
# DATA_DIR = os.path.join(os.getcwd(), "uploaded_files")
#######################################
# # ------------------ ACCOUNT & ADMIN ------------------
# with tabs[2]:
#     st.header("Account & Admin")

#     st.subheader("Account info")
#     if user:
#         st.json(user)
#     else:
#         st.warning("⚠️ No user logged in")
#     st.markdown("---")

#     # Admin features
#     if user and user.get('is_admin'):
#         st.subheader("Admin: view all uploads")
#         cur.execute("""
#             SELECT u.id as user_id, u.username, a.filename, a.uploaded_at, a.field_name, a.filepath
#             FROM users u 
#             JOIN uploads a ON u.id = a.user_id 
#             ORDER BY a.uploaded_at DESC 
#             LIMIT 500
#         """)
#         allrows = cur.fetchall()

#         if allrows:
#             adf = pd.DataFrame(allrows, columns=['user_id','username','filename','uploaded_at','field_name','filepath'])
#             st.dataframe(adf)

#             if st.button("Download combined CSV of all uploads"):
#                 frames = []
#                 for fp in adf['filepath']:
#                     if os.path.exists(fp):
#                         frames.append(pd.read_csv(fp))
#                 if frames:
#                     combined = pd.concat(frames, ignore_index=True)
#                     buf = BytesIO()
#                     combined.to_csv(buf, index=False)
#                     buf.seek(0)
#                     st.download_button(
#                         "Download combined CSV (final)",
#                         data=buf,
#                         file_name='combined_all_users.csv'
#                     )
#                 else:
#                     st.warning("No files found on disk")
#         else:
#             st.info("No uploads yet")
#     else:
#         st.info("You are not an admin. Admins can view/download uploads.")

# st.sidebar.markdown('---')
# if user and user.get("is_admin"):
#     st.sidebar.write(f"Local DB: {DB_PATH}")

# # Directory where files are stored
# DATA_DIR = os.path.join(os.getcwd(), "uploaded_files")
#######################################
import logging

# ------------------ LOGGING SETUP ------------------
LOG_FILE = "app_errors.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s"
)



# ------------------ ACCOUNT & ADMIN ------------------
with tabs[2]:
    st.header("Account & Admin")

    st.subheader("Account info")
    if user:
        st.json(user)
    else:
        st.warning("⚠️ No user logged in")
    st.markdown("---")

    # Admin features
    if user and user.get('is_admin'):
        st.subheader("🔑 Admin: Manage All Uploads")
        # your existing code for managing uploads...

        st.markdown("---")

        # 🔹 Admin Signup Form
        with st.form("admin_signup_form"):
            st.subheader("👑 Create New Admin")
            new_admin_user = st.text_input("Admin Username")
            new_admin_email = st.text_input("Admin Email")
            new_admin_pass = st.text_input("Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")
            submit_admin = st.form_submit_button("Create Admin")

            if submit_admin:
                if new_admin_pass != confirm_pass:
                    st.error("❌ Passwords do not match.")
                elif not new_admin_user or not new_admin_email or not new_admin_pass:
                    st.warning("⚠️ Please fill all fields.")
                else:
                    result = create_admin(new_admin_user, new_admin_pass, new_admin_email)
                    if result["status"] == "success":
                        st.success(result["message"])
                    else:
                        st.error(result["message"])
    else:
        st.info("You are not an admin. Admins can view/download uploads.")

########################################
# # ------------------ ACCOUNT & ADMIN ------------------
# with tabs[2]:
#     st.header("Account & Admin")

#     st.subheader("Account info")
#     if user:
#         st.json(user)
#     else:
#         st.warning("⚠️ No user logged in")
#     st.markdown("---")

#     # Admin features
#     if user and user.get('is_admin'):
#         st.subheader("🔑 Admin: Manage All Uploads")

#         try:
#             # Get all users
#             cur.execute("SELECT id, username FROM users ORDER BY username ASC")
#             all_users = cur.fetchall()
#         except Exception as e:
#             logging.error(f"Failed to fetch users: {e}", exc_info=True)
#             all_users = []

#         if all_users:
#             # User filter dropdown
#             usernames = {u[1]: u[0] for u in all_users}
#             selected_user = st.selectbox("👤 Select a user", list(usernames.keys()))

#             # Fetch that user’s uploads
#             cur.execute(
#                 "SELECT filename, uploaded_at, field_name, filepath FROM uploads WHERE user_id=? ORDER BY uploaded_at DESC",
#                 (usernames[selected_user],)
#             )
#             user_uploads = cur.fetchall()

#             if user_uploads:
#                 df_user = pd.DataFrame(user_uploads, columns=['Filename', 'Uploaded At', 'Field', 'Filepath'])
#                 st.dataframe(df_user[['Filename', 'Uploaded At', 'Field']])

#                 # Pick a file
#                 file_choice = st.selectbox("📂 Select a file to view", df_user['Filename'])

#                 # Show preview or allow download
#                 chosen_row = df_user[df_user['Filename'] == file_choice].iloc[0]
#                 filepath = chosen_row['Filepath']

#                 if os.path.exists(filepath):
#                     try:
#                         df_preview = pd.read_csv(filepath)
#                         st.write("📊 Preview of selected file:")
#                         st.dataframe(df_preview.head(20))  # show first 20 rows

#                         # Allow download
#                         with open(filepath, "rb") as f:
#                             st.download_button(
#                                 label="⬇️ Download file",
#                                 data=f,
#                                 file_name=chosen_row['Filename'],
#                                 mime="text/csv"
#                             )
#                     except Exception as e:
#                         logging.error(f"Failed to read file {filepath}: {e}", exc_info=True)
#                         st.error("⚠️ Could not open this file.")
#                 else:
#                     st.error("⚠️ File not found on disk.")
#             else:
#                 st.info("This user has no uploads.")
#         else:
#             st.info("No users found in system.")


#################
# # ------------------ ACCOUNT & ADMIN ------------------
# with tabs[2]:
#     st.header("Account & Admin")

#     st.subheader("Account info")
#     if user:
#         st.json(user)
#     else:
#         st.warning("⚠️ No user logged in")
#     st.markdown("---")

#     # Admin features
#     if user and user.get('is_admin'):
#         st.subheader("Admin: view all uploads")

#         try:
#             cur.execute("""
#                 SELECT u.id as user_id, u.username, a.filename, a.uploaded_at, a.field_name, a.filepath
#                 FROM users u 
#                 JOIN uploads a ON u.id = a.user_id 
#                 ORDER BY a.uploaded_at DESC 
#                 LIMIT 500
#             """)
#             allrows = cur.fetchall()
#         except Exception as e:
#             logging.error(f"Database query failed: {e}", exc_info=True)
#             st.error("⚠️ Could not load uploads. Please try again later.")
#             allrows = []

#         if allrows:
#             adf = pd.DataFrame(allrows, columns=['user_id','username','filename','uploaded_at','field_name','filepath'])
#             st.dataframe(adf)

#             if st.button("Download combined CSV of all uploads"):
#                 frames = []
#                 for fp in adf['filepath']:
#                     if os.path.exists(fp):
#                         try:
#                             frames.append(pd.read_csv(fp))
#                         except Exception as e:
#                             logging.error(f"Failed to read file {fp}: {e}", exc_info=True)
#                             st.warning(f"⚠️ Skipped a corrupted/missing file.")
#                 if frames:
#                     combined = pd.concat(frames, ignore_index=True)
#                     buf = BytesIO()
#                     combined.to_csv(buf, index=False)
#                     buf.seek(0)
#                     st.download_button(
#                         "Download combined CSV (final)",
#                         data=buf,
#                         file_name='combined_all_users.csv'
#                     )
#                 else:
#                     st.warning("No files found on disk")
#         else:
#             st.info("No uploads yet")
#     else:
#         st.info("You are not an admin. Admins can view/download uploads.")

# st.sidebar.markdown('---')
# if user and user.get("is_admin"):
#     st.sidebar.write(f"Local DB: {DB_PATH}")

# # Directory where files are stored
# DATA_DIR = os.path.join(os.getcwd(), "uploaded_files")




# # ------------------ RECENT UPLOADS TAB ------------------
# with tabs[3]:
#     st.header("🕒 Recent Uploads")

#     if user["is_admin"]:
#         cur.execute("""
#             SELECT u.filename, u.field_name, u.notes, u.uploaded_at, usr.username
#             FROM uploads u
#             JOIN users usr ON u.user_id = usr.id
#             ORDER BY u.uploaded_at DESC
#             LIMIT 10
#         """)
#         rows = cur.fetchall()

#         if rows:
#             st.subheader("All Users' Recent Uploads (Admin)")
#             for idx, r in enumerate(rows):
#                 col1, col2, col3, col4 = st.columns([2,3,2,2])
#                 col1.write(r[0])   # filename
#                 col2.write(r[4])   # uploaded_by
#                 col3.write(r[3])   # uploaded_at

#                 # Use separate keys
#                 confirm_state_key = f"admin_confirm_state_{idx}"
#                 delete_button_key = f"admin_delete_btn_{idx}"
#                 confirm_button_key = f"admin_confirm_btn_{idx}"

#                 if st.session_state.get(confirm_state_key, False):
#                     if col4.button("❌ Confirm Delete", key=confirm_button_key):
#                         # Delete from DB
#                         cur.execute("DELETE FROM uploads WHERE filename=?", (r[0],))
#                         conn.commit()

#                         # Delete from disk
#                         file_path = os.path.join(DATA_DIR, r[0])
#                         if os.path.exists(file_path):
#                             os.remove(file_path)

#                         st.warning(f"Admin deleted {r[0]}")
#                         st.session_state[confirm_state_key] = False
#                         st.experimental_rerun()
#                 else:
#                     if col4.button("🗑️ Delete", key=delete_button_key):
#                         st.session_state[confirm_state_key] = True
#                         st.error(f"⚠️ Click confirm to delete {r[0]}")
#         else:
#             st.info("No recent uploads in the system.")

#     else:
#         cur.execute(
#             "SELECT filename, field_name, notes, uploaded_at FROM uploads WHERE user_id = ? ORDER BY uploaded_at DESC LIMIT 5", 
#             (user['id'],) if user else (-1,)

#         )
#         rows = cur.fetchall()

#         if rows:
#             st.subheader("My Recent Uploads")
#             for idx, r in enumerate(rows):
#                 col1, col2, col3 = st.columns([3,3,2])
#                 col1.write(r[0])   # filename
#                 col2.write(r[3])   # uploaded_at

#                 # Use separate keys
#                 confirm_state_key = f"user_confirm_state_{idx}"
#                 delete_button_key = f"user_delete_btn_{idx}"
#                 confirm_button_key = f"user_confirm_btn_{idx}"

#                 if st.session_state.get(confirm_state_key, False):
#                     if col3.button("❌ Confirm Delete", key=confirm_button_key):
#                         # Delete from DB
#                         cur.execute("DELETE FROM uploads WHERE filename=? AND user_id=?", (r[0], user['id']))
#                         conn.commit()

#                         # Delete from disk
#                         file_path = os.path.join(DATA_DIR, r[0])
#                         if os.path.exists(file_path):
#                             os.remove(file_path)

#                         st.success(f"Deleted {r[0]}")
#                         st.session_state[confirm_state_key] = False
#                         st.experimental_rerun()
#                 else:
#                     if col3.button("🗑️ Delete", key=delete_button_key):
#                         st.session_state[confirm_state_key] = True
#                         st.error(f"⚠️ Click confirm to delete {r[0]}")
#         else:
#             st.info("You have no recent uploads.")
#################################################################
# ------------------ RECENT UPLOADS TAB ------------------
with tabs[3]:
    st.header("🕒 Recent Uploads")

    if user:  # ✅ only run if someone is logged in
        if user.get("is_admin"):
            cur.execute("""
                SELECT u.filename, u.field_name, u.notes, u.uploaded_at, usr.username
                FROM uploads u
                JOIN users usr ON u.user_id = usr.id
                ORDER BY u.uploaded_at DESC
                LIMIT 10
            """)
            rows = cur.fetchall()

            if rows:
                st.subheader("All Users' Recent Uploads (Admin)")
                for idx, r in enumerate(rows):
                    col1, col2, col3, col4 = st.columns([2,3,2,2])
                    col1.write(r[0])   # filename
                    col2.write(r[4])   # uploaded_by
                    col3.write(r[3])   # uploaded_at

                    # Use separate keys
                    confirm_state_key = f"admin_confirm_state_{idx}"
                    delete_button_key = f"admin_delete_btn_{idx}"
                    confirm_button_key = f"admin_confirm_btn_{idx}"

                    if st.session_state.get(confirm_state_key, False):
                        if col4.button("❌ Confirm Delete", key=confirm_button_key):
                            # Delete from DB
                            cur.execute("DELETE FROM uploads WHERE filename=?", (r[0],))
                            conn.commit()

                            # Delete from disk
                            file_path = os.path.join(DATA_DIR, r[0])
                            if os.path.exists(file_path):
                                os.remove(file_path)

                            st.warning(f"Admin deleted {r[0]}")
                            st.session_state[confirm_state_key] = False
                            st.experimental_rerun()
                    else:
                        if col4.button("🗑️ Delete", key=delete_button_key):
                            st.session_state[confirm_state_key] = True
                            st.error(f"⚠️ Click confirm to delete {r[0]}")
            else:
                st.info("No recent uploads in the system.")

        else:  # normal user
            cur.execute(
                "SELECT filename, field_name, notes, uploaded_at FROM uploads WHERE user_id = ? ORDER BY uploaded_at DESC LIMIT 5", 
                (user['id'],)
            )
            rows = cur.fetchall()

            if rows:
                st.subheader("My Recent Uploads")
                for idx, r in enumerate(rows):
                    col1, col2, col3 = st.columns([3,3,2])
                    col1.write(r[0])   # filename
                    col2.write(r[3])   # uploaded_at

                    # Use separate keys
                    confirm_state_key = f"user_confirm_state_{idx}"
                    delete_button_key = f"user_delete_btn_{idx}"
                    confirm_button_key = f"user_confirm_btn_{idx}"

                    if st.session_state.get(confirm_state_key, False):
                        if col3.button("❌ Confirm Delete", key=confirm_button_key):
                            # Delete from DB
                            cur.execute("DELETE FROM uploads WHERE filename=? AND user_id=?", (r[0], user['id']))
                            conn.commit()

                            # Delete from disk
                            file_path = os.path.join(DATA_DIR, r[0])
                            if os.path.exists(file_path):
                                os.remove(file_path)

                            st.success(f"Deleted {r[0]}")
                            st.session_state[confirm_state_key] = False
                            st.experimental_rerun()
                    else:
                        if col3.button("🗑️ Delete", key=delete_button_key):
                            st.session_state[confirm_state_key] = True
                            st.error(f"⚠️ Click confirm to delete {r[0]}")
            else:
                st.info("You have no recent uploads.")
    else:
        st.info("Please log in to view recent uploads.")  # ✅ safe when not logged in





# # ------------------ SAVED FILES TAB ------------------
# with tabs[4]:  # adjust index depending on your layout query language
#     st.header("📂 Saved Files")

#     if user["is_admin"]:
#         cur.execute("""
#             SELECT u.filepath, u.filename, u.field_name, u.notes, u.uploaded_at, usr.username
#             FROM uploads u
#             JOIN users usr ON u.user_id = usr.id
#             ORDER BY u.uploaded_at DESC
#         """)
#         rows = cur.fetchall()

#         if rows:
#             df_files = pd.DataFrame(rows, columns=["Filepath", "Filename", "Field", "Notes", "Uploaded At", "Uploaded By"])
#             for idx, row in df_files.iterrows():
#                 st.markdown(f"**{row['Filename']}** — Field: {row['Field']} | By: {row['Uploaded By']} | {row['Uploaded At']}")
#                 try:
#                     with open(row["Filepath"], "rb") as f:
#                         st.download_button(
#                             label="⬇️ Download",
#                             data=f,
#                             file_name=row["Filename"],
#                             mime="text/csv",
#                             key=f"download_{idx}"
#                         )
#                 except FileNotFoundError:
#                     st.error(f"File {row['Filename']} not found on disk.")
#         else:
#             st.info("No files saved in the system yet.")
#     else:
#         cur.execute(
#             "SELECT filepath, filename, field_name, notes, uploaded_at FROM uploads WHERE user_id = ? ORDER BY uploaded_at DESC",
#             (user['id'],) if user else (-1,)

#         )
#         rows = cur.fetchall()

#         if rows:
#             df_files = pd.DataFrame(rows, columns=["Filepath", "Filename", "Field", "Notes", "Uploaded At"])
#             for idx, row in df_files.iterrows():
#                 st.markdown(f"**{row['Filename']}** — Field: {row['Field']} | {row['Uploaded At']}")
#                 try:
#                     with open(row["Filepath"], "rb") as f:
#                         st.download_button(
#                             label="⬇️ Download",
#                             data=f,
#                             file_name=row["Filename"],
#                             mime="text/csv",
#                             key=f"user_download_{idx}"
#                         )
#                 except FileNotFoundError:
#                     st.error(f"File {row['Filename']} not found on disk.")
#         else:
#             st.info("You have not saved any files yet.")
#############################
# ------------------ SAVED FILES TAB ------------------
with tabs[4]:  # adjust index depending on your layout
    st.header("📂 Saved Files")

    if user:  # ✅ only run if logged in
        if user.get("is_admin"):
            cur.execute("""
                SELECT u.filepath, u.filename, u.field_name, u.notes, u.uploaded_at, usr.username
                FROM uploads u
                JOIN users usr ON u.user_id = usr.id
                ORDER BY u.uploaded_at DESC
            """)
            rows = cur.fetchall()

            if rows:
                df_files = pd.DataFrame(
                    rows, 
                    columns=["Filepath", "Filename", "Field", "Notes", "Uploaded At", "Uploaded By"]
                )
                for idx, row in df_files.iterrows():
                    st.markdown(f"**{row['Filename']}** — Field: {row['Field']} | By: {row['Uploaded By']} | {row['Uploaded At']}")
                    try:
                        with open(row["Filepath"], "rb") as f:
                            st.download_button(
                                label="⬇️ Download",
                                data=f,
                                file_name=row["Filename"],
                                mime="text/csv",
                                key=f"download_{idx}"
                            )
                    except FileNotFoundError:
                        st.error(f"File {row['Filename']} not found on disk.")
            else:
                st.info("No files saved in the system yet.")
        else:  # normal user
            cur.execute(
                "SELECT filepath, filename, field_name, notes, uploaded_at FROM uploads WHERE user_id = ? ORDER BY uploaded_at DESC",
                (user['id'],)
            )
            rows = cur.fetchall()

            if rows:
                df_files = pd.DataFrame(
                    rows, 
                    columns=["Filepath", "Filename", "Field", "Notes", "Uploaded At"]
                )
                for idx, row in df_files.iterrows():
                    st.markdown(f"**{row['Filename']}** — Field: {row['Field']} | {row['Uploaded At']}")
                    try:
                        with open(row["Filepath"], "rb") as f:
                            st.download_button(
                                label="⬇️ Download",
                                data=f,
                                file_name=row["Filename"],
                                mime="text/csv",
                                key=f"user_download_{idx}"
                            )
                    except FileNotFoundError:
                        st.error(f"File {row['Filename']} not found on disk.")
            else:
                st.info("You have not saved any files yet.")
    else:
        st.info("Please log in to view saved files.")  # ✅ safe when not logged in



import sqlite3
import hashlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

DB_PATH = "production_app.db"

# --- Password Hashing ---
def hash_password(password: str) -> str:
    salt = "static_salt_change_me"  # should match your app’s user signup
    return hashlib.sha256((salt + password).encode()).hexdigest()

# --- Email Notification with Brevo ---
def send_email(to_email: str, subject: str, body: str) -> bool:
    sender_email = "hafsatuxy@gmail.com"          # must be verified in Brevo
    brevo_login = "96cf99001@smtp-brevo.com"      # from Brevo dashboard
    brevo_password = "OKnmRy6V7fUY509I"           # SMTP key from Brevo
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

        return True
    except Exception as e:
        print("❌ Failed to send email:", e)
        return False

# --- Create Admin Function ---
def create_admin(username: str, password: str, email: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Ensure users table exists
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        email TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0,
        is_verified INTEGER DEFAULT 0,
        created_at TEXT
    )
    """)

    hashed_pw = hash_password(password)

    try:
        cur.execute("""
           INSERT INTO users (username, password_hash, is_admin, email, is_verified, created_at)
           VALUES (?, ?, ?, ?, ?, ?)
        """, (username, hashed_pw, 1, email, 1, datetime.utcnow().isoformat()))

        conn.commit()

        # Send notification email
        subject = "🔐 New Admin Account Created"
        body = f"""
Hello {username},

✅ Your admin account has been created successfully.

📧 Email: {email}
👑 Role: Admin

You now have full admin privileges.
        """
        send_email(email, subject, body)

        return {"status": "success", "message": f"Admin '{username}' created and notified via email."}

    except sqlite3.IntegrityError:
        return {"status": "error", "message": f"Username '{username}' already exists."}

    finally:
        conn.close()


























