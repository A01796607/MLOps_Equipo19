"""
DVC Integration for Data Versioning with S3 Storage.

This module provides DVC integration for versioning data files and models
with S3 as remote storage. It handles pulling data from S3, pushing data
to S3, and managing DVC configuration.
"""
import subprocess
from pathlib import Path
from typing import Optional, Dict
from loguru import logger

# Try to import DVC (optional dependency)
try:
    import dvc.api
    from dvc.repo import Repo
    DVC_AVAILABLE = True
except ImportError:
    DVC_AVAILABLE = False
    logger.warning("DVC not available. Data versioning features will be disabled.")


class DVCManager:
    """
    Manager class for DVC data versioning with S3 storage.
    """
    
    def __init__(self):
        """Initialize DVC Manager."""
        self._dvc_repo = None
        if DVC_AVAILABLE:
            try:
                self._dvc_repo = Repo()
                logger.debug("DVC repository initialized")
            except Exception as e:
                logger.warning(f"DVC repository not found: {e}")
                self._dvc_repo = None
    
    @property
    def dvc_repo(self):
        """Lazy initialization of DVC repo."""
        if not DVC_AVAILABLE:
            return None
        if self._dvc_repo is None:
            try:
                self._dvc_repo = Repo()
            except Exception:
                pass
        return self._dvc_repo
    
    @staticmethod
    def setup_s3_remote(
        bucket_name: str,
        access_key_id: str,
        secret_access_key: str,
        region: str = "us-east-1",
        remote_name: str = "s3-storage",
        remote_path: str = "dvc-data",
        set_as_default: bool = True
    ) -> bool:
        """
        Configure DVC to use S3 as remote storage.
        
        Args:
            bucket_name: S3 bucket name (without s3:// prefix)
            access_key_id: AWS access key ID
            secret_access_key: AWS secret access key
            region: AWS region (default: us-east-1)
            remote_name: Name for the DVC remote (default: s3-storage)
            remote_path: Path within bucket for DVC data (default: dvc-data)
            set_as_default: Whether to set this as the default remote
            
        Returns:
            True if configuration was successful, False otherwise
        """
        if not DVC_AVAILABLE:
            logger.error("DVC is not installed. Install with: pip install 'dvc[s3]'")
            return False
        
        try:
            repo_root = Path.cwd()
            
            # Initialize DVC if not already initialized
            if not (repo_root / ".dvc").exists():
                logger.info("Initializing DVC repository...")
                subprocess.check_call(
                    ['dvc', 'init'],
                    cwd=repo_root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            # Check if remote already exists
            try:
                existing_remotes = subprocess.check_output(
                    ['dvc', 'remote', 'list'],
                    cwd=repo_root,
                    stderr=subprocess.DEVNULL
                ).decode().strip().split('\n')
                
                if remote_name in existing_remotes:
                    logger.info(f"Remote '{remote_name}' already exists. Removing it first...")
                    subprocess.check_call(
                        ['dvc', 'remote', 'remove', remote_name],
                        cwd=repo_root,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
            except subprocess.CalledProcessError:
                pass
            
            # Add S3 remote
            s3_url = f"s3://{bucket_name}/{remote_path}"
            logger.info(f"Adding DVC remote '{remote_name}' pointing to {s3_url}...")
            
            subprocess.check_call(
                ['dvc', 'remote', 'add', remote_name, s3_url],
                cwd=repo_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Configure credentials
            subprocess.check_call(
                ['dvc', 'remote', 'modify', remote_name, 'access_key_id', access_key_id],
                cwd=repo_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            subprocess.check_call(
                ['dvc', 'remote', 'modify', remote_name, 'secret_access_key', secret_access_key],
                cwd=repo_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            subprocess.check_call(
                ['dvc', 'remote', 'modify', remote_name, 'region', region],
                cwd=repo_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Set as default if requested
            if set_as_default:
                subprocess.check_call(
                    ['dvc', 'remote', 'default', remote_name],
                    cwd=repo_root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            logger.success(f"DVC S3 remote configured successfully!")
            logger.info(f"Remote name: {remote_name}")
            logger.info(f"S3 URL: {s3_url}")
            logger.info(f"Region: {region}")
            logger.info(f"Default remote: {'Yes' if set_as_default else 'No'}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error configuring DVC S3 remote: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error configuring DVC S3 remote: {e}")
            return False
    
    def ensure_versioned(self, data_path: Path, data_type: str = "dataset") -> bool:
        """
        Ensure data is versioned with DVC. If not, add it to DVC tracking.
        
        Args:
            data_path: Path to data file/directory to version
            data_type: Type of data (for logging)
            
        Returns:
            True if data is now versioned, False otherwise
        """
        if not DVC_AVAILABLE:
            logger.warning("DVC not available. Cannot version data.")
            return False
        
        try:
            repo_root = Path.cwd()
            data_path = Path(data_path)
            
            # Add to DVC (this updates if already tracked, or adds if new)
            logger.info(f"Ensuring {data_type} is versioned with DVC...")
            subprocess.check_call(
                ['dvc', 'add', str(data_path)],
                cwd=repo_root
            )
            
            logger.success(f"{data_type} successfully versioned/updated with DVC")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error adding {data_type} to DVC: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error versioning {data_type}: {e}")
            return False
    
    def push_to_s3(self) -> bool:
        """
        Push all DVC-tracked data to S3 remote.
        
        Returns:
            True if push was successful, False otherwise
        """
        if not DVC_AVAILABLE:
            logger.error("DVC not available. Cannot push to S3.")
            return False
        
        try:
            logger.info("Pushing DVC-tracked data to S3...")
            subprocess.check_call(
                ['dvc', 'push'],
                cwd=Path.cwd()
            )
            logger.success("Data successfully pushed to S3")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error pushing data to S3: {e}")
            return False
    
    def pull_from_s3(self) -> bool:
        """
        Pull all DVC-tracked data from S3 remote.
        
        Returns:
            True if pull was successful, False otherwise
        """
        if not DVC_AVAILABLE:
            logger.error("DVC not available. Cannot pull from S3.")
            return False
        
        try:
            logger.info("Pulling DVC-tracked data from S3...")
            subprocess.check_call(
                ['dvc', 'pull'],
                cwd=Path.cwd()
            )
            logger.success("Data successfully pulled from S3")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error pulling data from S3: {e}")
            return False

