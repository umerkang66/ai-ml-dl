# 🚀 Deployment Guide: Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and log in using your GitHub account.
2. Click the **"New app"** button in the upper-right corner.
3. In the deployment form, configure the fields:
   - **Repository**: Choose `your-username/your-repo-name` from the dropdown list.
   - **Branch**: Select `main`.
   - **Main file path**: Enter `app.py`.
4. Click **"Deploy!"**.

---

## ⚙️ Advanced Configuration (Optional)

Streamlit Cloud allows customizing the theme. You can create a file at `.streamlit/config.toml` to enforce a dark/light mode default:

```toml
[theme]
primaryColor = "#00c6ff"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#1f2937"
textColor = "#f0f2f6"
font = "sans serif"
```

---

> [!NOTE]
> Streamlit Community Cloud will automatically build your app's environment by running `pip install -r requirements.txt`. The process might take 2-3 minutes during the first deploy.
