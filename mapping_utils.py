"""
Утилиты для создания маппинга между кодами и параметрами
"""
import json
import os
from typing import Dict, List, Any


class ParameterMapper:
    """Класс для маппинга параметров на короткие коды"""
    
    def __init__(self, mapping_file_path: str):
        """
        Args:
            mapping_file_path: Путь к JSON файлу для хранения маппинга
        """
        self.mapping_file_path = mapping_file_path
        self.code_to_params: Dict[int, Dict[str, float]] = {}
        self.params_to_code: Dict[str, int] = {}
        self.next_code = 1
        
        # Загружаем существующий маппинг, если файл есть
        self._load_mapping()
    
    def _load_mapping(self):
        """Загружает маппинг из JSON файла"""
        if os.path.exists(self.mapping_file_path):
            try:
                with open(self.mapping_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Конвертируем строковые ключи обратно в int
                    self.code_to_params = {int(k): v for k, v in data.get('code_to_params', {}).items()}
                    self.params_to_code = data.get('params_to_code', {})
                    self.next_code = data.get('next_code', 1)
            except Exception as e:
                print(f"Warning: Could not load mapping file: {e}")
    
    def _save_mapping(self):
        """Сохраняет маппинг в JSON файл"""
        # Создаем директорию, если её нет
        os.makedirs(os.path.dirname(self.mapping_file_path), exist_ok=True)
        
        with open(self.mapping_file_path, 'w', encoding='utf-8') as f:
            json.dump({
                'code_to_params': self.code_to_params,
                'params_to_code': self.params_to_code,
                'next_code': self.next_code
            }, f, indent=2, ensure_ascii=False)
    
    def get_or_create_code(self, params: Dict[str, float]) -> int:
        """
        Получает код для параметров или создаёт новый
        
        Args:
            params: Словарь параметров
            
        Returns:
            int: Код (1, 2, 3, ...)
        """
        # Создаем ключ для params_to_code
        params_key = json.dumps(params, sort_keys=True)
        
        if params_key in self.params_to_code:
            return self.params_to_code[params_key]
        
        # Создаем новый код
        code = self.next_code
        self.code_to_params[code] = params
        self.params_to_code[params_key] = code
        self.next_code += 1
        
        # Сохраняем после каждого добавления
        self._save_mapping()
        
        return code
    
    def get_params(self, code: int) -> Dict[str, float]:
        """Получает параметры по коду"""
        return self.code_to_params.get(code, {})
    
    def get_code_info(self, code: int) -> str:
        """Возвращает строковое представление кода с параметрами"""
        params = self.get_params(code)
        if params:
            params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            return f"Code {code}: {params_str}"
        return f"Code {code}: not found"