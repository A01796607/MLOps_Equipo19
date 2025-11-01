#!/usr/bin/env python3
"""
Script to configure DVC with S3 remote storage.

This script sets up DVC to use AWS S3 as remote storage for versioned data.
Run this script once to configure your DVC repository with S3.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dvcS3 import DVCManager
from loguru import logger

def main():
    """Configure DVC with S3."""
    import os
    
    # AWS Credentials - Load from environment variables
    # Set these in your environment or use AWS profile instead
    ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
    SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    REGION = "us-east-2"

    print(f"ACCESS_KEY_ID: {ACCESS_KEY_ID}")
    print(f"SECRET_ACCESS_KEY: {SECRET_ACCESS_KEY}")
    
    # S3 Configuration
    # Bucket path: s3://itesm-mna/202502-equipo19
    BUCKET_NAME = "itesm-mna"
    BUCKET_PATH = "202502-equipo19"
    
    logger.info("=" * 60)
    logger.info("Configuring DVC with S3 Remote Storage")
    logger.info("=" * 60)
    
    # Setup DVC S3 remote
    # Note: If you have AWS profile configured, you can use it instead of credentials
    # If ACCESS_KEY_ID is empty, setup will use AWS profile from ~/.aws/credentials
    success = DVCManager.setup_s3_remote(
        bucket_name=BUCKET_NAME,
        access_key_id=ACCESS_KEY_ID,
        secret_access_key=SECRET_ACCESS_KEY,
        region=REGION,
        remote_name="team_remote",
        remote_path=BUCKET_PATH,
        set_as_default=True
    )
    
    # Optionally configure AWS profile (if using AWS profiles)
    import subprocess
    from pathlib import Path
    try:
        subprocess.check_call(
            ['dvc', 'remote', 'modify', 's3-storage', 'profile', 'equipo19'],
            cwd=Path.cwd(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        logger.info("Configured to use AWS profile: equipo19")
    except:
        pass
    
    if success:
        logger.success("\n✅ DVC S3 configuration completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Create the S3 bucket if it doesn't exist:")
        logger.info(f"   aws s3 mb s3://{BUCKET_NAME} --region {REGION}")
        logger.info("\n2. Add data to DVC tracking:")
        logger.info("   dvc add data/raw/")
        logger.info("\n3. Push data to S3:")
        logger.info("   dvc push")
        logger.info("\n4. Or use the DVCManager methods:")
        logger.info("   from src.dvc import DVCManager")
        logger.info("   manager = DVCManager()")
        logger.info("   manager.push_to_s3()")
    else:
        logger.error("\n❌ Failed to configure DVC with S3.")
        logger.error("Please check your credentials and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
