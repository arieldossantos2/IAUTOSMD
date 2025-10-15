-- --------------------------------------------------------
-- Banco de Dados: smt_inspection_new
-- --------------------------------------------------------

CREATE DATABASE IF NOT EXISTS `smt_inspection_new`;
USE `smt_inspection_new`;

-- --------------------------------------------------------
-- Estrutura da tabela `components`
-- --------------------------------------------------------
CREATE TABLE `components` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `product_id` int(11) NOT NULL,
  `name` varchar(255) NOT NULL,
  `x` int(11) NOT NULL,
  `y` int(11) NOT NULL,
  `width` int(11) NOT NULL,
  `height` int(11) NOT NULL,
  `package_id` int(11) DEFAULT NULL,
  `rotation` int(11) NOT NULL DEFAULT 0,
  `inspection_mask` longtext DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------
-- Estrutura da tabela `inspections`
-- --------------------------------------------------------
CREATE TABLE `inspections` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `product_id` int(11) NOT NULL,
  `produced_image_path` varchar(255) DEFAULT NULL,
  `overall_status` enum('OK','FAIL','IN_PROGRESS') DEFAULT 'IN_PROGRESS',
  `total_ok` int(11) DEFAULT 0,
  `total_fail` int(11) DEFAULT 0,
  `inspection_timestamp` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------
-- Estrutura da tabela `inspection_results`
-- --------------------------------------------------------
CREATE TABLE `inspection_results` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `inspection_id` int(11) NOT NULL,
  `component_id` int(11) NOT NULL,
  `status` enum('OK','FAIL') NOT NULL,
  `rotation` varchar(10) DEFAULT NULL,
  `displacement_x` int(11) DEFAULT NULL,
  `displacement_y` int(11) DEFAULT NULL,
  `metrics` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL CHECK (json_valid(`metrics`)),
  `roi_golden_path` varchar(255) DEFAULT NULL,
  `roi_produced_path` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------
-- Estrutura da tabela `packages`
-- --------------------------------------------------------
CREATE TABLE `packages` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `description` text DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------
-- Estrutura da tabela `products`
-- --------------------------------------------------------
CREATE TABLE `products` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `revision` varchar(100) DEFAULT NULL,
  `golden_image_path` varchar(255) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------
-- Estrutura da tabela `users`
-- --------------------------------------------------------
CREATE TABLE `users` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `username` varchar(100) NOT NULL,
  `password_hash` varchar(255) NOT NULL,
  `role` enum('ADMIN','OPERATOR') DEFAULT 'OPERATOR',
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------
-- Chaves estrangeiras
-- --------------------------------------------------------
ALTER TABLE `components`
  ADD CONSTRAINT `components_product_fk` FOREIGN KEY (`product_id`) REFERENCES `products` (`id`) ON DELETE CASCADE;

ALTER TABLE `inspections`
  ADD CONSTRAINT `inspections_product_fk` FOREIGN KEY (`product_id`) REFERENCES `products` (`id`) ON DELETE CASCADE;

ALTER TABLE `inspection_results`
  ADD CONSTRAINT `inspection_results_inspection_fk` FOREIGN KEY (`inspection_id`) REFERENCES `inspections` (`id`) ON DELETE CASCADE,
  ADD CONSTRAINT `inspection_results_component_fk` FOREIGN KEY (`component_id`) REFERENCES `components` (`id`) ON DELETE CASCADE;

CREATE TABLE `training_samples` (
  `id` INT(11) NOT NULL AUTO_INCREMENT,
  `product_id` INT(11) DEFAULT NULL,
  `component_id` INT(11) DEFAULT NULL,
  `golden_base64` LONGTEXT DEFAULT NULL,   -- imagem padrão (de referência)
  `produced_base64` LONGTEXT NOT NULL,     -- imagem produzida (real)
  `label` VARCHAR(50) NOT NULL,            -- rótulo (ex: OK, FAIL, etc.)
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  CONSTRAINT `fk_training_product`
    FOREIGN KEY (`product_id`) REFERENCES `products` (`id`) ON DELETE CASCADE,
  CONSTRAINT `fk_training_component`
    FOREIGN KEY (`component_id`) REFERENCES `components` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

DROP USER IF EXISTS 'flaskuser'@'localhost';
CREATE USER 'flaskuser'@'localhost' IDENTIFIED BY '123456';
GRANT ALL PRIVILEGES ON smt_inspection_new.* TO 'flaskuser'@'localhost';
FLUSH PRIVILEGES;
