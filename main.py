import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import re

warnings.filterwarnings('ignore')

def clean_text(text):
    """Очистка текста: нижний регистр, удаление лишних пробелов и спецсимволов."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\sа-яё]', ' ', text) # Оставляем только буквы и пробелы
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def validate_output(answer_df: pd.DataFrame, benchmark_items: pd.DataFrame):
    """Строгая валидация выходных данных."""
    valid_item_ids = set(benchmark_items['item_id'].astype(str).unique())
    
    for idx, row in answer_df.iterrows():
        q_id = str(row['query_id'])
        answer_str = str(row['answer'])
        
        assert len(q_id) == 16, f"Неверная длина query_id: {q_id}"
        
        item_ids = answer_str.split()
        assert len(item_ids) <= 50, f"Слишком много item_id для запроса {q_id}: {len(item_ids)}"
        assert len(item_ids) == len(set(item_ids)), f"Найдены дубликаты item_id для запроса {q_id}"
        
        for item_id in item_ids:
            assert len(item_id) == 16, f"Неверная длина item_id: {item_id}"
            assert re.match(r'^[0-9a-f]{16}$', item_id), f"item_id не hex: {item_id}"
            assert item_id in valid_item_ids, f"item_id {item_id} нет в benchmark_items"

    print("   [OK] Все проверки формата успешно пройдены!")

def main():
    print("1. Загрузка данных...")
    benchmark_queries = pd.read_parquet('benchmark_queries.parquet')
    benchmark_items = pd.read_parquet('benchmark_items.parquet')
    
    print(f"   Загружено запросов: {len(benchmark_queries)}")
    print(f"   Загружено объявлений: {len(benchmark_items)}")

    print("\n2. Предобработка данных...")
    benchmark_items['item_id'] = benchmark_items['item_id'].astype(str).str.strip()
    benchmark_queries['query_id'] = benchmark_queries['query_id'].astype(str).str.strip()
    
    # Очищаем и заполняем пропуски
    benchmark_items['item_title_raw'] = benchmark_items['item_title_raw'].fillna('').apply(clean_text)
    benchmark_items['item_description_raw'] = benchmark_items['item_description_raw'].fillna('').apply(clean_text)
    benchmark_queries['search_query'] = benchmark_queries['search_query'].fillna('').apply(clean_text)
    
    # Создаем единый текстовый признак для объявления
    # Умножаем заголовок на 3, чтобы он имел больший вес при TF-IDF, чем описание
    items_texts = (benchmark_items['item_title_raw'] * 3 + " " + benchmark_items['item_description_raw']).tolist()

    print("\n3. Векторизация корпуса (TF-IDF с символьными n-граммами)...")
    # analyzer='char_wb' создает n-граммы внутри слов, что идеально для русского языка 
    # и устойчиво к опечаткам и морфологии (например, "телевизор" и "телевизоров").
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 4),
        max_features=50000,
        dtype=np.float32
    )
    
    print("   Обучение векторайзера и трансформация объявлений...")
    item_tfidf = vectorizer.fit_transform(items_texts)

    print("\n4. Поиск кандидатов (TF-IDF + эвристические бусты)...")
    results = []
    batch_size = 100  # Обрабатываем запросы батчами для скорости
    
    for start_idx in range(0, len(benchmark_queries), batch_size):
        end_idx = min(start_idx + batch_size, len(benchmark_queries))
        batch_queries = benchmark_queries.iloc[start_idx:end_idx]
        
        # Векторизуем запросы (используем тот же vectorizer, только transform)
        batch_query_texts = batch_queries['search_query'].tolist()
        query_tfidf = vectorizer.transform(batch_query_texts)
        
        # Считаем косинусное сходство (быстрая операция над разреженными матрицами)
        batch_similarities = cosine_similarity(query_tfidf, item_tfidf)
        
        for i in range(len(batch_queries)):
            row = batch_queries.iloc[i]
            sim_scores = batch_similarities[i]
            
            # ШАГ А: Берем топ-1000 по тексту (этого достаточно для применения бустов)
            # np.argpartition работает за O(N), очень быстро
            top_1000_idx = np.argpartition(sim_scores, -1000)[-1000:]
            
            candidate_items = benchmark_items.iloc[top_1000_idx]
            final_scores = sim_scores[top_1000_idx].copy()
            
            # ШАГ Б: Эвристические бусты (КРИТИЧЕСКИ ВАЖНО для Авито)
            # Буст за совпадение категории: если пользователь ищет в "Ремонт", 
            # мы должны сильно предпочесть объявления из "Ремонт".
            if pd.notna(row['search_category']):
                cat_mask = candidate_items['item_category_id'] == row['search_category']
                final_scores[cat_mask] += 0.5  # Сильный буст
                
            # Буст за совпадение локации
            if pd.notna(row['search_location_id']):
                loc_mask = candidate_items['item_location_id'] == row['search_location_id']
                final_scores[loc_mask] += 0.3
                
            # ШАГ В: Финальная сортировка и выбор топ-50
            top_50_local_idx = np.argsort(final_scores)[-50:][::-1]
            top_50_item_ids = candidate_items.iloc[top_50_local_idx]['item_id'].tolist()
            
            results.append({
                'query_id': row['query_id'],
                'answer': ' '.join(top_50_item_ids)
            })
            
        if end_idx % 500 == 0:
            print(f"   Обработано {end_idx} из {len(benchmark_queries)} запросов...")

    print("\n5. Формирование и валидация результата...")
    answer_df = pd.DataFrame(results)
    
    assert len(answer_df) == len(benchmark_queries), "Количество строк не совпадает!"
    assert list(answer_df.columns) == ['query_id', 'answer'], "Неверные имена колонок!"
    
    validate_output(answer_df, benchmark_items)
    
    answer_df.to_csv('answer.csv', index=False, encoding='utf-8')
    print("   Готово! Результат успешно сохранен в файл 'answer.csv'")

if __name__ == '__main__':
    main()