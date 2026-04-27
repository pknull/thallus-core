//! Node identity -- Ed25519 keypair for signing node-authored messages.
//!
//! Each node has one long-term Ed25519 identity used for signing task results
//! and capability profiles. Public IDs use SSB wire format: `@<base64-pubkey>.ed25519`.
//!
//! Key storage: raw 32-byte file (`secret.key`).

use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::{CoreError, Result};

/// Full identity with private key (local only).
pub struct Identity {
    signing_key: SigningKey,
}

impl Clone for Identity {
    fn clone(&self) -> Self {
        Self {
            signing_key: SigningKey::from_bytes(&self.signing_key.to_bytes()),
        }
    }
}

/// Public identity for wire format: `@<base64-pubkey>.ed25519`
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PublicId(pub String);

impl Identity {
    /// Generate a new random identity.
    pub fn generate() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        Self { signing_key }
    }

    /// Get the verifying (public) key.
    pub fn verifying_key(&self) -> VerifyingKey {
        self.signing_key.verifying_key()
    }

    /// Get the public ID in wire format.
    pub fn public_id(&self) -> PublicId {
        PublicId::from_verifying_key(&self.verifying_key())
    }

    /// Sign a message and return base64-encoded signature.
    pub fn sign(&self, message: &[u8]) -> String {
        let signature = self.signing_key.sign(message);
        B64.encode(signature.to_bytes())
    }

    /// Sign a hash (hex string) and return base64-encoded signature.
    pub fn sign_hash(&self, hash: &str) -> String {
        self.sign(hash.as_bytes())
    }

    /// Save private key bytes to file (unencrypted).
    pub fn save(&self, path: &Path) -> Result<()> {
        let dir = path.parent().ok_or_else(|| CoreError::Config {
            reason: "invalid key path".into(),
        })?;
        std::fs::create_dir_all(dir)?;
        std::fs::write(path, self.signing_key.to_bytes())?;

        // Set restrictive file permissions on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            std::fs::set_permissions(path, perms)?;
        }

        Ok(())
    }

    /// Load private key from file.
    pub fn load(path: &Path) -> Result<Self> {
        validate_private_key_permissions(path)?;
        let bytes = std::fs::read(path).map_err(|_| CoreError::IdentityNotFound {
            path: path.display().to_string(),
        })?;
        let key_bytes: [u8; 32] = bytes.try_into().map_err(|_| CoreError::InvalidKeypair {
            reason: "expected 32 bytes".into(),
        })?;
        let signing_key = SigningKey::from_bytes(&key_bytes);
        Ok(Self { signing_key })
    }

    /// Load or generate identity at the given directory.
    pub fn load_or_generate(identity_dir: &Path) -> Result<Self> {
        let key_path = identity_dir.join("secret.key");
        if key_path.exists() {
            Self::load(&key_path)
        } else {
            let identity = Self::generate();
            identity.save(&key_path)?;
            // Also save public key for convenience
            let pub_path = identity_dir.join("public.key");
            std::fs::write(pub_path, identity.public_id().0.as_bytes())?;
            tracing::info!(public_id = %identity.public_id(), "generated new identity");
            Ok(identity)
        }
    }
}

fn validate_private_key_permissions(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let mode = std::fs::metadata(path)?.permissions().mode() & 0o777;
        if mode & 0o077 != 0 {
            return Err(CoreError::Config {
                reason: format!(
                    "private key {} has insecure permissions {:o}; expected owner-only access",
                    path.display(),
                    mode
                ),
            });
        }
    }

    Ok(())
}

impl PublicId {
    /// Validate that a string is a well-formed public ID.
    pub fn is_valid_format(s: &str) -> bool {
        // @<44 base64 chars>.ed25519 = 1 + 44 + 8 = 53
        if s.len() != 53 || !s.starts_with('@') || !s.ends_with(".ed25519") {
            return false;
        }
        let b64_part = &s[1..45];
        B64.decode(b64_part).is_ok_and(|bytes| bytes.len() == 32)
    }

    /// Create from a verifying (public) key.
    pub fn from_verifying_key(vk: &VerifyingKey) -> Self {
        let encoded = B64.encode(vk.as_bytes());
        Self(format!("@{}.ed25519", encoded))
    }

    /// Parse the base64 public key bytes from the wire format.
    pub fn to_verifying_key(&self) -> Result<VerifyingKey> {
        let inner = self
            .0
            .strip_prefix('@')
            .and_then(|s| s.strip_suffix(".ed25519"))
            .ok_or_else(|| CoreError::InvalidKeypair {
                reason: format!("invalid public ID format: {}", self.0),
            })?;
        let bytes = B64.decode(inner).map_err(|e| CoreError::InvalidKeypair {
            reason: format!("base64 decode failed: {e}"),
        })?;
        let key_bytes: [u8; 32] = bytes.try_into().map_err(|_| CoreError::InvalidKeypair {
            reason: "expected 32 bytes after decode".into(),
        })?;
        VerifyingKey::from_bytes(&key_bytes).map_err(|e| CoreError::InvalidKeypair {
            reason: format!("invalid ed25519 public key: {e}"),
        })
    }

    /// Verify a signature against this public key.
    pub fn verify(&self, message: &[u8], signature_b64: &str) -> Result<bool> {
        let vk = self.to_verifying_key()?;
        let sig_bytes = B64
            .decode(signature_b64)
            .map_err(|e| CoreError::InvalidKeypair {
                reason: format!("signature base64 decode failed: {e}"),
            })?;
        let sig_bytes: [u8; 64] = sig_bytes
            .try_into()
            .map_err(|_| CoreError::InvalidKeypair {
                reason: "expected 64 bytes for signature".into(),
            })?;
        let signature = Signature::from_bytes(&sig_bytes);
        Ok(vk.verify(message, &signature).is_ok())
    }
}

impl std::fmt::Display for PublicId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_and_roundtrip_public_id() {
        let identity = Identity::generate();
        let pub_id = identity.public_id();

        assert!(pub_id.0.starts_with('@'));
        assert!(pub_id.0.ends_with(".ed25519"));

        let vk = pub_id.to_verifying_key().unwrap();
        assert_eq!(vk, identity.verifying_key());
    }

    #[test]
    fn sign_and_verify() {
        let identity = Identity::generate();
        let message = b"test message";
        let signature = identity.sign(message);

        let pub_id = identity.public_id();
        assert!(pub_id.verify(message, &signature).unwrap());
    }

    #[test]
    fn save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let key_path = dir.path().join("secret.key");

        let identity = Identity::generate();
        identity.save(&key_path).unwrap();

        let loaded = Identity::load(&key_path).unwrap();
        assert_eq!(identity.verifying_key(), loaded.verifying_key());
    }

    #[test]
    fn load_or_generate_creates_new() {
        let dir = tempfile::tempdir().unwrap();
        let identity_dir = dir.path().join("identity");

        let id1 = Identity::load_or_generate(&identity_dir).unwrap();
        let id2 = Identity::load_or_generate(&identity_dir).unwrap();

        // Same key loaded both times
        assert_eq!(id1.verifying_key(), id2.verifying_key());
    }

    #[cfg(unix)]
    #[test]
    fn load_rejects_group_readable_private_key() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let key_path = dir.path().join("secret.key");

        let identity = Identity::generate();
        identity.save(&key_path).unwrap();
        std::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(0o644)).unwrap();

        match Identity::load(&key_path) {
            Ok(_) => panic!("insecure key permissions must fail"),
            Err(err) => assert!(err.to_string().contains("insecure permissions")),
        }
    }

    #[test]
    fn valid_format_check() {
        let identity = Identity::generate();
        let pub_id = identity.public_id();
        assert!(PublicId::is_valid_format(&pub_id.0));
        assert!(!PublicId::is_valid_format("not-valid"));
        assert!(!PublicId::is_valid_format("@short.ed25519"));
    }
}
