from cryptography.fernet import Fernet
import os

class APIVault:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_vault()
        return cls._instance

    def _init_vault(self):
        self.key_file = ".vaultkey"
        if not os.path.exists(self.key_file):
            with open(self.key_file, 'wb') as f:
                f.write(Fernet.generate_key())
        with open(self.key_file, 'rb') as f:
            self.cipher = Fernet(f.read())

    def encrypt(self, secret: str) -> bytes:
        return self.cipher.encrypt(secret.encode())

    def decrypt(self, encrypted: bytes) -> str:
        return self.cipher.decrypt(encrypted).decode()

# Contoh penggunaan:
vault = APIVault()
encrypted_key = vault.encrypt("your_okx_api_key_here")  # Simpan di .env
decrypted_key = vault.decrypt(encrypted_key)  # Untuk dipakai di runtime
