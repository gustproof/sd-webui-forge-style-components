import launch

if not launch.is_installed("scikit-learn"):
    launch.run_pip("install scikit-learn>=1", "requirements for Style Components")