-- =====================================================
-- ТЕСТОВЫЙ СКРИПТ ДЛЯ ЗАПОЛНЕНИЯ ДАННЫХ
-- ДЛЯ КОНКРЕТНОГО ПОЛЬЗОВАТЕЛЯ (ФИНАЛЬНАЯ ВЕРСИЯ)
-- =====================================================

-- ВАЖНО: Замените email на реальный email вашего пользователя
DO $$
DECLARE
    target_user_email VARCHAR := '220103378@stu.sdu.edu.kz'; -- <-- ЗАМЕНИТЕ НА СВОЙ EMAIL
    target_user_sid VARCHAR;
    v_upload_sid VARCHAR;
    category_count INTEGER;
    product_count INTEGER;
    -- Переменные для статистики
    stats_products INTEGER;
    stats_warehouse_items INTEGER;
    stats_store_items INTEGER;
    stats_sales INTEGER;
    stats_revenue NUMERIC;
BEGIN
    -- Проверяем существование пользователя
    SELECT sid INTO target_user_sid
    FROM "user"
    WHERE email = target_user_email AND is_verified = true;

    IF target_user_sid IS NULL THEN
        RAISE EXCEPTION 'Пользователь с email % не найден или не верифицирован', target_user_email;
    END IF;

    RAISE NOTICE 'Найден пользователь: % (sid: %)', target_user_email, target_user_sid;

    -- 1. СОЗДАЕМ КАТЕГОРИИ (если их еще нет)
    INSERT INTO category (id, sid, name)
    VALUES
        (gen_random_uuid(), substr('cat_' || md5(random()::text || 'dairy' || clock_timestamp()::text), 1, 22), 'Молочные продукты'),
        (gen_random_uuid(), substr('cat_' || md5(random()::text || 'bakery' || clock_timestamp()::text), 1, 22), 'Хлебобулочные изделия'),
        (gen_random_uuid(), substr('cat_' || md5(random()::text || 'drinks' || clock_timestamp()::text), 1, 22), 'Напитки'),
        (gen_random_uuid(), substr('cat_' || md5(random()::text || 'fruits' || clock_timestamp()::text), 1, 22), 'Овощи и фрукты'),
        (gen_random_uuid(), substr('cat_' || md5(random()::text || 'meat' || clock_timestamp()::text), 1, 22), 'Мясо и птица'),
        (gen_random_uuid(), substr('cat_' || md5(random()::text || 'sweets' || clock_timestamp()::text), 1, 22), 'Кондитерские изделия'),
        (gen_random_uuid(), substr('cat_' || md5(random()::text || 'frozen' || clock_timestamp()::text), 1, 22), 'Замороженные продукты'),
        (gen_random_uuid(), substr('cat_' || md5(random()::text || 'grocery' || clock_timestamp()::text), 1, 22), 'Бакалея')
    ON CONFLICT (name) DO NOTHING;

    SELECT COUNT(*) INTO category_count FROM category;
    RAISE NOTICE 'Всего категорий в базе: %', category_count;

    -- 2. СОЗДАЕМ ПРОДУКТЫ С УЧЕТОМ ПОЛЬЗОВАТЕЛЕЙ
    -- Сначала создаем общие продукты (без привязки к пользователю)
    WITH categories AS (
        SELECT sid, name FROM category
    ),
    products_to_insert AS (
        SELECT
            gen_random_uuid() as id,
            substr('prd_' || md5(random()::text || p.name || clock_timestamp()::text), 1, 22) as sid,
            c.sid as category_sid,
            p.name,
            p.barcode,
            p.unit as default_unit,
            p.price::float as default_price,
            'KZT'::currency as currency,
            p.storage_duration,
            'DAY'::storagedurationtype as storage_duration_type
        FROM (
            VALUES
                -- Молочные продукты
                ('Молоко Простоквашино 3.2% 1л', '4607002991234', 'шт', 480, 5, 'day', 'Молочные продукты'),
                ('Кефир Био-С 2.5% 1л', '4607002991235', 'шт', 420, 7, 'day', 'Молочные продукты'),
                ('Сметана 20% 200г', '4607002991236', 'шт', 320, 10, 'day', 'Молочные продукты'),
                ('Йогурт Данон клубника', '4607002991237', 'шт', 180, 14, 'day', 'Молочные продукты'),
                ('Творог 5% 200г', '4607002991238', 'шт', 280, 7, 'day', 'Молочные продукты'),

                -- Хлебобулочные
                ('Хлеб Бородинский', '4607002991239', 'шт', 200, 3, 'day', 'Хлебобулочные изделия'),
                ('Батон нарезной', '4607002991240', 'шт', 150, 2, 'day', 'Хлебобулочные изделия'),
                ('Булочка с маком 5шт', '4607002991241', 'уп', 250, 2, 'day', 'Хлебобулочные изделия'),
                ('Лаваш армянский', '4607002991242', 'шт', 120, 5, 'day', 'Хлебобулочные изделия'),

                -- Напитки
                ('Кока-Кола 1.5л', '4607002991243', 'шт', 450, 180, 'day', 'Напитки'),
                ('Сок Rich апельсин 1л', '4607002991244', 'шт', 580, 365, 'day', 'Напитки'),
                ('Вода BonAqua 1.5л', '4607002991245', 'шт', 180, 730, 'day', 'Напитки'),
                ('Чай Lipton зеленый 100п', '4607002991246', 'уп', 1200, 730, 'day', 'Напитки'),
                ('Кофе Jacobs растворимый 95г', '4607002991247', 'шт', 2800, 730, 'day', 'Напитки'),

                -- Овощи и фрукты
                ('Яблоки Семеренко', '4607002991248', 'кг', 580, 30, 'day', 'Овощи и фрукты'),
                ('Бананы', '4607002991249', 'кг', 690, 7, 'day', 'Овощи и фрукты'),
                ('Помидоры розовые', '4607002991250', 'кг', 890, 10, 'day', 'Овощи и фрукты'),
                ('Огурцы', '4607002991251', 'кг', 750, 7, 'day', 'Овощи и фрукты'),
                ('Картофель', '4607002991252', 'кг', 180, 60, 'day', 'Овощи и фрукты'),

                -- Мясо и птица
                ('Куриное филе охл.', '4607002991253', 'кг', 2400, 5, 'day', 'Мясо и птица'),
                ('Фарш говяжий', '4607002991254', 'кг', 3200, 3, 'day', 'Мясо и птица'),
                ('Свинина шейка', '4607002991255', 'кг', 2800, 5, 'day', 'Мясо и птица'),
                ('Колбаса Докторская', '4607002991256', 'кг', 2100, 30, 'day', 'Мясо и птица'),

                -- Кондитерские изделия
                ('Шоколад Алёнка 100г', '4607002991257', 'шт', 380, 365, 'day', 'Кондитерские изделия'),
                ('Печенье Юбилейное', '4607002991258', 'уп', 280, 180, 'day', 'Кондитерские изделия'),
                ('Конфеты Рафаэлло', '4607002991259', 'уп', 1850, 180, 'day', 'Кондитерские изделия'),

                -- Замороженные продукты
                ('Пельмени Цезарь 800г', '4607002991260', 'уп', 980, 180, 'day', 'Замороженные продукты'),
                ('Мороженое пломбир', '4607002991261', 'шт', 320, 365, 'day', 'Замороженные продукты'),

                -- Бакалея
                ('Макароны Barilla 500г', '4607002991262', 'уп', 480, 730, 'day', 'Бакалея'),
                ('Рис длиннозерный 800г', '4607002991263', 'уп', 520, 730, 'day', 'Бакалея'),
                ('Масло подсолнечное 1л', '4607002991264', 'шт', 680, 730, 'day', 'Бакалея'),
                ('Сахар песок 1кг', '4607002991265', 'уп', 420, 1095, 'day', 'Бакалея')
        ) AS p(name, barcode, unit, price, storage_duration, storage_duration_type, category_name)
        JOIN categories c ON c.name = p.category_name
    )
    INSERT INTO product (id, sid, category_sid, name, barcode, default_unit, default_price, currency, storage_duration, storage_duration_type)
    SELECT id, sid, category_sid, name, barcode, default_unit, default_price, currency, storage_duration, storage_duration_type
    FROM products_to_insert
    ON CONFLICT (barcode) DO NOTHING;

    SELECT COUNT(*) INTO product_count FROM product;
    RAISE NOTICE 'Всего продуктов в базе: %', product_count;

    -- 3. СОЗДАЕМ ЗАГРУЗКУ
    v_upload_sid := substr('upl_' || md5(target_user_sid || NOW()::text), 1, 22);

    INSERT INTO upload (id, sid, user_sid, file_name, uploaded_at, rows_imported)
    VALUES (
        gen_random_uuid(),
        v_upload_sid,
        target_user_sid,
        'test_import_' || to_char(NOW(), 'YYYYMMDD_HH24MI') || '.xlsx',
        NOW() - INTERVAL '120 days',
        product_count
    );

    RAISE NOTICE 'Создана загрузка: %', v_upload_sid;

    -- 4. СОЗДАЕМ ТОВАРЫ НА СКЛАДЕ И В МАГАЗИНЕ С ПОЛНОЙ ИСТОРИЕЙ
    WITH date_series AS (
        SELECT generate_series(
            CURRENT_DATE - INTERVAL '119 days',
            CURRENT_DATE,
            INTERVAL '1 day'
        )::date as supply_date
    ),
    all_products AS (
        SELECT sid, name, default_price, storage_duration, barcode
        FROM product
    ),
    warehouse_inserts AS (
        INSERT INTO warehouseitem (id, sid, upload_sid, product_sid, batch_code, quantity, expire_date, received_at, status, urgency_level)
        SELECT
            gen_random_uuid(),
            substr('whi_' || md5(random()::text || p.sid || ds.supply_date || clock_timestamp()::text), 1, 22),
            v_upload_sid,
            p.sid,
            'BATCH' || to_char(ds.supply_date, 'YYYYMMDD'),
            CASE
                WHEN p.barcode IN ('4607002991234', '4607002991239') THEN 150 + (random() * 50)::int
                WHEN p.barcode IN ('4607002991243', '4607002991244', '4607002991245') THEN 100 + (random() * 30)::int
                WHEN p.barcode IN ('4607002991248', '4607002991249', '4607002991250') THEN 80 + (random() * 20)::int
                ELSE 60 + (random() * 20)::int
            END,
            ds.supply_date + (p.storage_duration || ' days')::interval,
            ds.supply_date,
            'MOVED'::warehouseitemstatus,
            'NORMAL'::urgencylevel
        FROM all_products p
        CROSS JOIN date_series ds
        WHERE
            -- Генерируем поставки для всех товаров с разной периодичностью
            (p.barcode = '4607002991234' AND EXTRACT(DAY FROM ds.supply_date)::integer % 2 = 0) OR  -- Молоко каждые 2 дня
            (p.barcode = '4607002991239' AND EXTRACT(DAY FROM ds.supply_date)::integer % 1 = 0) OR  -- Хлеб каждый день
            (p.barcode = '4607002991243' AND EXTRACT(DAY FROM ds.supply_date)::integer % 5 = 0) OR  -- Кола каждые 5 дней
            (p.barcode = '4607002991244' AND EXTRACT(DAY FROM ds.supply_date)::integer % 7 = 0) OR  -- Сок каждую неделю
            (p.barcode = '4607002991245' AND EXTRACT(DAY FROM ds.supply_date)::integer % 3 = 0) OR  -- Вода каждые 3 дня
            (p.barcode = '4607002991248' AND EXTRACT(DAY FROM ds.supply_date)::integer % 4 = 0) OR  -- Яблоки каждые 4 дня
            (p.barcode = '4607002991249' AND EXTRACT(DAY FROM ds.supply_date)::integer % 3 = 0) OR  -- Бананы каждые 3 дня
            (p.barcode = '4607002991250' AND EXTRACT(DAY FROM ds.supply_date)::integer % 5 = 0) OR  -- Помидоры каждые 5 дней
            (p.barcode = '4607002991253' AND EXTRACT(DAY FROM ds.supply_date)::integer % 4 = 0) OR  -- Курица каждые 4 дня
            (p.barcode = '4607002991257' AND EXTRACT(DAY FROM ds.supply_date)::integer % 10 = 0) OR -- Шоколад каждые 10 дней
            (p.barcode = '4607002991260' AND EXTRACT(DAY FROM ds.supply_date)::integer % 7 = 0) OR  -- Пельмени каждую неделю
            (p.barcode = '4607002991262' AND EXTRACT(DAY FROM ds.supply_date)::integer % 14 = 0)    -- Макароны каждые 2 недели
        RETURNING sid, product_sid, quantity, received_at, (SELECT default_price FROM product WHERE sid = product_sid) as price
    )
    INSERT INTO storeitem (id, sid, warehouse_item_sid, quantity, price, moved_at, status)
    SELECT
        gen_random_uuid(),
        substr('sti_' || md5(random()::text || wi.sid || clock_timestamp()::text), 1, 22),
        wi.sid,
        wi.quantity,
        wi.price * 1.35,
        wi.received_at + INTERVAL '2 hours',
        'ACTIVE'::storeitemstatus
    FROM warehouse_inserts wi;

    RAISE NOTICE 'Создана история поставок и перемещений в магазин';

    -- 5. ГЕНЕРИРУЕМ ПОДРОБНУЮ ИСТОРИЮ ПРОДАЖ ДЛЯ ВСЕХ ТОВАРОВ
    WITH store_items AS (
        SELECT
            si.sid,
            si.price,
            si.quantity as initial_quantity,
            si.moved_at,
            p.name,
            p.barcode,
            p.sid as product_sid,
            wi.received_at,
            wi.expire_date
        FROM storeitem si
        JOIN warehouseitem wi ON si.warehouse_item_sid = wi.sid
        JOIN product p ON wi.product_sid = p.sid
        JOIN upload u ON wi.upload_sid = u.sid
        WHERE u.user_sid = target_user_sid
    ),
    sales_generator AS (
        SELECT
            si.sid as store_item_sid,
            si.price,
            si.name,
            si.product_sid,
            gs.sale_date,
            EXTRACT(DOW FROM gs.sale_date) as day_of_week,
            EXTRACT(MONTH FROM gs.sale_date) as month,
            -- Генерируем количество продаж с учетом типа товара, дня недели и сезонности
            GREATEST(1, LEAST(
                si.initial_quantity / GREATEST(1, (CURRENT_DATE - si.received_at::date)::numeric),
                CASE
                    -- Молоко - стабильные ежедневные продажи
                    WHEN si.barcode = '4607002991234' THEN
                        CASE
                            WHEN EXTRACT(DOW FROM gs.sale_date) IN (0, 6) THEN 30 + (random() * 15)::int
                            ELSE 20 + (random() * 10)::int
                        END
                    -- Хлеб - высокие продажи каждый день
                    WHEN si.barcode = '4607002991239' THEN
                        CASE
                            WHEN EXTRACT(DOW FROM gs.sale_date) IN (0, 6) THEN 50 + (random() * 20)::int
                            ELSE 35 + (random() * 15)::int
                        END
                    -- Напитки - сезонность и выходные
                    WHEN si.barcode IN ('4607002991243', '4607002991244', '4607002991245') THEN
                        CASE
                            WHEN EXTRACT(MONTH FROM gs.sale_date) IN (6, 7, 8) THEN 25 + (random() * 15)::int
                            WHEN EXTRACT(DOW FROM gs.sale_date) IN (0, 6) THEN 20 + (random() * 10)::int
                            ELSE 10 + (random() * 8)::int
                        END
                    -- Овощи и фрукты - сезонные колебания
                    WHEN si.barcode IN ('4607002991248', '4607002991249', '4607002991250') THEN
                        CASE
                            WHEN EXTRACT(MONTH FROM gs.sale_date) IN (7, 8, 9) THEN 20 + (random() * 10)::int
                            ELSE 10 + (random() * 8)::int
                        END
                    -- Мясо - больше продаж в выходные
                    WHEN si.barcode = '4607002991253' THEN
                        CASE
                            WHEN EXTRACT(DOW FROM gs.sale_date) IN (5, 6, 0) THEN 15 + (random() * 10)::int
                            ELSE 8 + (random() * 5)::int
                        END
                    -- Кондитерские изделия - импульсные покупки
                    WHEN si.barcode = '4607002991257' THEN
                        (5 + (random() * 8)::int) * (CASE WHEN random() > 0.6 THEN 2 ELSE 1 END)
                    -- Замороженные продукты - зимой больше
                    WHEN si.barcode = '4607002991260' THEN
                        CASE
                            WHEN EXTRACT(MONTH FROM gs.sale_date) IN (11, 12, 1, 2) THEN 10 + (random() * 5)::int
                            ELSE 5 + (random() * 3)::int
                        END
                    -- Остальные товары
                    ELSE
                        8 + (random() * 5)::int
                END
            ))::int as quantity
        FROM store_items si
        CROSS JOIN generate_series(
            si.received_at::date,
            LEAST(CURRENT_DATE - INTERVAL '1 day', si.received_at::date + INTERVAL '118 days'),
            INTERVAL '1 day'
        ) AS gs(sale_date)
        WHERE
            -- Продаем только активные товары и до истечения срока годности
            (si.expire_date IS NULL OR gs.sale_date <= si.expire_date - INTERVAL '1 day')
    )
    INSERT INTO sale (id, sid, store_item_sid, sold_qty, sold_price, sold_at, cashier_sid)
    SELECT
        gen_random_uuid(),
        substr('sal_' || md5(random()::text || sg.store_item_sid || sg.sale_date || clock_timestamp()::text), 1, 22),
        sg.store_item_sid,
        sg.quantity,
        sg.price,
        sg.sale_date + (
            CASE
                WHEN sg.day_of_week IN (0, 6) THEN
                    (INTERVAL '10 hours' + (random() * INTERVAL '10 hours'))
                ELSE
                    (INTERVAL '8 hours' + (random() * INTERVAL '12 hours'))
            END
        ),
        target_user_sid
    FROM sales_generator sg
    WHERE sg.quantity > 0;

    RAISE NOTICE 'Создана подробная история продаж';

    -- 6. ОБНОВЛЯЕМ ОСТАТКИ В МАГАЗИНЕ
    UPDATE storeitem si
    SET quantity = GREATEST(0, si.quantity - COALESCE(sold.total_sold, 0))
    FROM (
        SELECT
            s.store_item_sid,
            SUM(s.sold_qty) as total_sold
        FROM sale s
        GROUP BY s.store_item_sid
    ) sold
    WHERE si.sid = sold.store_item_sid
    AND EXISTS (
        SELECT 1 FROM warehouseitem wi
        JOIN upload u ON wi.upload_sid = u.sid
        WHERE wi.sid = si.warehouse_item_sid
        AND u.user_sid = target_user_sid
    );

    -- 7. СОЗДАЕМ АКТУАЛЬНЫЕ ТОВАРЫ НА СКЛАДЕ ДЛЯ ТЕКУЩЕЙ РАБОТЫ
    INSERT INTO warehouseitem (id, sid, upload_sid, product_sid, batch_code, quantity, expire_date, received_at, status, urgency_level)
    SELECT
        gen_random_uuid(),
        substr('whi_curr_' || md5(random()::text || p.sid || CURRENT_DATE || clock_timestamp()::text), 1, 22),
        v_upload_sid,
        p.sid,
        'BATCH' || to_char(CURRENT_DATE, 'YYYYMMDD'),
        CASE
            WHEN p.barcode = '4607002991234' THEN 100
            WHEN p.barcode = '4607002991239' THEN 80
            WHEN p.barcode = '4607002991243' THEN 60
            WHEN p.barcode = '4607002991244' THEN 50
            WHEN p.barcode = '4607002991245' THEN 70
            WHEN p.barcode = '4607002991248' THEN 40
            WHEN p.barcode = '4607002991249' THEN 45
            WHEN p.barcode = '4607002991250' THEN 35
            WHEN p.barcode = '4607002991253' THEN 30
            ELSE 25
        END,
        CURRENT_DATE + (p.storage_duration || ' days')::interval,
        CURRENT_DATE,
        'IN_STOCK'::warehouseitemstatus,
        'NORMAL'::urgencylevel
    FROM product p
    WHERE p.barcode IN (
        '4607002991234', '4607002991239', '4607002991243', '4607002991244',
        '4607002991245', '4607002991248', '4607002991249', '4607002991250',
        '4607002991253', '4607002991257', '4607002991260', '4607002991262'
    );

    RAISE NOTICE 'Добавлены текущие товары на склад для работы с системой';

    -- 8. СОБИРАЕМ СТАТИСТИКУ
    SELECT
        COUNT(DISTINCT p.sid),
        COUNT(DISTINCT wi.sid),
        COUNT(DISTINCT si.sid),
        COUNT(DISTINCT s.sid),
        COALESCE(SUM(s.sold_qty * s.sold_price), 0)
    INTO
        stats_products,
        stats_warehouse_items,
        stats_store_items,
        stats_sales,
        stats_revenue
    FROM sale s
    JOIN storeitem si ON s.store_item_sid = si.sid
    JOIN warehouseitem wi ON si.warehouse_item_sid = wi.sid
    JOIN product p ON wi.product_sid = p.sid
    JOIN upload u ON wi.upload_sid = u.sid
    WHERE u.user_sid = target_user_sid;

    -- Выводим итоговую статистику
    RAISE NOTICE '=== ИТОГОВАЯ СТАТИСТИКА ДЛЯ ПОЛЬЗОВАТЕЛЯ % ===', target_user_email;
    RAISE NOTICE 'Уникальных продуктов: %', stats_products;
    RAISE NOTICE 'Записей на складе: %', stats_warehouse_items;
    RAISE NOTICE 'Записей в магазине: %', stats_store_items;
    RAISE NOTICE 'Продаж: %', stats_sales;
    RAISE NOTICE 'Общая выручка: %', stats_revenue;
    RAISE NOTICE '';
    RAISE NOTICE 'Данные успешно созданы!';
    RAISE NOTICE 'Теперь вы можете использовать систему прогнозирования';

EXCEPTION
    WHEN OTHERS THEN
        RAISE NOTICE 'Произошла ошибка: %', SQLERRM;
        RAISE NOTICE 'SQLSTATE: %', SQLSTATE;
        RAISE;
END $$;

-- Проверка созданных данных
SELECT
    'Статистика для пользователя:' as info,
    u.email,
    COUNT(DISTINCT p.sid) as products,
    COUNT(DISTINCT wi.sid) as warehouse_items,
    COUNT(DISTINCT si.sid) as store_items,
    COUNT(DISTINCT s.sid) as sales,
    COALESCE(SUM(s.sold_qty * s.sold_price), 0)::numeric(10,2) as total_revenue
FROM "user" u
LEFT JOIN upload up ON u.sid = up.user_sid
LEFT JOIN warehouseitem wi ON up.sid = wi.upload_sid
LEFT JOIN product p ON wi.product_sid = p.sid
LEFT JOIN storeitem si ON wi.sid = si.warehouse_item_sid
LEFT JOIN sale s ON si.sid = s.store_item_sid
WHERE u.email = '220103378@stu.sdu.edu.kz'  -- <-- ЗАМЕНИТЕ НА СВОЙ EMAIL
GROUP BY u.email;

-- Проверка данных для прогнозирования
SELECT
    p.name as product_name,
    COUNT(DISTINCT DATE(s.sold_at)) as sale_days,
    SUM(s.sold_qty) as total_quantity,
    MIN(DATE(s.sold_at)) as first_sale_date,
    MAX(DATE(s.sold_at)) as last_sale_date,
    AVG(s.sold_qty)::numeric(10,2) as avg_daily_sales
FROM sale s
JOIN storeitem si ON s.store_item_sid = si.sid
JOIN warehouseitem wi ON si.warehouse_item_sid = wi.sid
JOIN product p ON wi.product_sid = p.sid
JOIN upload u ON wi.upload_sid = u.sid
JOIN "user" usr ON u.user_sid = usr.sid
WHERE usr.email = '220103378@stu.sdu.edu.kz'  -- <-- ЗАМЕНИТЕ НА СВОЙ EMAIL
GROUP BY p.sid, p.name
HAVING COUNT(DISTINCT DATE(s.sold_at)) >= 14
ORDER BY sale_days DESC;