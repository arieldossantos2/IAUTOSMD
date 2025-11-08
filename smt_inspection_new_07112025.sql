-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Tempo de geração: 07/11/2025 às 18:06
-- Versão do servidor: 10.4.32-MariaDB
-- Versão do PHP: 8.0.30

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Banco de dados: `smt_inspection_new`
--

-- --------------------------------------------------------

--
-- Estrutura para tabela `components`
--

CREATE TABLE `components` (
  `id` int(11) NOT NULL,
  `product_id` int(11) NOT NULL,
  `name` varchar(255) NOT NULL,
  `x` int(11) NOT NULL,
  `y` int(11) NOT NULL,
  `width` int(11) NOT NULL,
  `height` int(11) NOT NULL,
  `package_id` int(11) DEFAULT NULL,
  `rotation` int(11) NOT NULL DEFAULT 0,
  `inspection_mask` longtext DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Despejando dados para a tabela `components`
--

INSERT INTO `components` (`id`, `product_id`, `name`, `x`, `y`, `width`, `height`, `package_id`, `rotation`, `inspection_mask`) VALUES
(1, 1, 'REP01', 199, 46, 93, 146, 1, 0, 'iVBORw0KGgoAAAANSUhEUgAAAF0AAACSCAAAAAACpJl+AAAB9UlEQVRoBbXBwREAMAzCMHv/oekEcPlUkp/kJ/lJfpKf5Cf5SX6Sn+Qn+Ul+kp/kJ/lJfpKf5Cf5SX6SLRxIIVs4kEK2cCCFbOFACtnCgRSyhQMpZAsHUsgWDqSQLRxIIVs4kEK2cCCFbOFACtnCgRSyhQMpZAsHUsgWDqSQLRxIIVs4kEK2cCCFbOFACtnCgRSyhQMpZAsHUsgWDqSQLRxIIVs4kEK2cCCFbOFACtnCgRSyhQMpZAsHUsgWDqSQLRxIIVs4kEK2cCCFbOFACtnCgRSyhQMpZAsHUsgWDqSQLRxIIVs4kEK2cCCFbOFACtnCgRSyhQMpZAsHUsgWDqSQLRxIIVs4kEK2cCCFbOFACtnCgRSyhQMpZAsHUsgWDqSQLRxIIVs4kEK2cCCFbOFACtnCgRSyhQMpZAsHUsgWDqSQLRxIIVs4kEK2cCCFbOFACtnCgRSyhQMpZAsHUsgWDqSQLRxIIVs4kEK2cCCFbOFACtnCgRSyhQMpZAsHUsgWDqSQLRxIIVs4kEK2cCCFbOFACtnCgRSyhQMpZAsHUsgWDqSQLRxIIVs4kEK2cCCFbOFACtnCgRTyk/wkP8lP8pP8JD/JT/KT/CQ/yU/yk/wkP8lP8pP8JD/JT/KT/CQ/yU/yk/wkP8lP8pP8JD/JT/LTA2zFXZPlWTTPAAAAAElFTkSuQmCC'),
(2, 1, 'COMP2', 356, 44, 89, 149, 1, 0, 'iVBORw0KGgoAAAANSUhEUgAAAFkAAACVCAAAAAAWSgm8AAAB+ElEQVRoBbXBwQ0AMBDCsGT/oekCCN2ntvwiv8gv8ov8Ir/IL/KL/CK/yC/yi/wiv8gv8ov8Ir/IL/KL/CK/yBAOpJMhHEgnQziQToZwIJ0M4UA6GcKBdDKEA+lkCAfSyRAOpJMhHEgnQziQToZwIJ0M4UA6GcKBdDKEA+lkCAfSyRAOpJMhHEgnQziQToZwIJ0M4UA6GcKBdDKEA+lkCAfSyRAOpJMhHEgnQziQToZwIJ0M4UA6GcKBdDKEA+lkCAfSyRAOpJMhHEgnQziQToZwIJ0M4UA6GcKBdDKEA+lkCAfSyRAOpJMhHEgnQziQToZwIJ0M4UA6GcKBdDKEA+lkCAfSyRAOpJMhHEgnQziQToZwIJ0M4UA6GcKBdDKEA+lkCAfSyRAOpJMhHEgnQziQToZwIJ0M4UA6GcKBdDKEA+lkCAfSyRAOpJMhHEgnQziQToZwIJ0M4UA6GcKBdDKEA+lkCAfSyRAOpJMhHEgnQziQToZwIJ0M4UA6GcKBdDKEA+lkCAfSyRAOpJMhHEgnQziQToZwIJ0M4UA6GcKBdDKEA+lkCAfSyRAOpJMhHEgnQziQToZwIJ38Ir/IL/KL/CK/yC/yi/wiv8gv8ov8Ir/IL/KL/CK/yC/yi/wiv8gv8ov8Ir/IL/KL/CK/yC/yi/wiv8gvD7KLXJbh7LvNAAAAAElFTkSuQmCC'),
(3, 2, 'C01', 294, 96, 66, 103, 2, 90, 'iVBORw0KGgoAAAANSUhEUgAAAEIAAABnCAAAAACRXsuxAAABNUlEQVRYCaXBsQ0AMAzDMOn/o93Ri4EMJeWbfJNv8k2+yTf5Jt/km3yTb/JNvsk3+Sbf5Jt8kwoHWaTCQRapcJBFKhxkkQoHWaTCQRapcJBFKhxkkQoHWaTCQRapcJBFKhxkkQoHWaTCQRapcJBFKhxkkQoHWaTCQRapcJBFKhxkkQoHWaTCQRapcJBFKhxkkQoHWaTCQRapcJBFKhxkkQoHWaTCQRapcJBFKhxkkQoHWaTCQRapcJBFKhxkkQoHWaTCQRapcJBFKhxkkQoHWaTCQRapcJBFKhxkkQoHWaTCQRapcJBFKhxkkQoHWaTCQRapcJBFKhxkkQoHWaTCQRapcJBFvsk3+Sbf5Jt8k2/yTb7JN/km3+SbfJNv8k2+yTf5Jt/km3yTb/JNvsk3+Sbf5Jt8ewy8N2gDprUzAAAAAElFTkSuQmCC'),
(4, 2, 'COMP2', 206, 49, 74, 134, 1, 0, NULL),
(5, 2, 'COMP3', 363, 50, 71, 133, 1, 0, NULL),
(6, 2, 'COMP4', 461, 50, 78, 135, 1, 0, NULL),
(7, 2, 'COMP5', 575, 52, 70, 132, 1, 0, NULL),
(8, 2, 'COMP6', 295, 227, 79, 130, 1, 0, NULL),
(9, 2, 'COMP7', 567, 227, 87, 131, 1, 0, NULL),
(10, 2, 'COMP8', 821, 226, 109, 138, 1, 0, NULL),
(11, 2, 'COMP9', 814, 48, 66, 133, 1, 0, NULL),
(12, 2, 'COMP10', 704, 47, 68, 132, 1, 0, NULL),
(13, 3, 'R01', 212, 44, 72, 142, 1, 0, 'iVBORw0KGgoAAAANSUhEUgAAAEgAAACOCAAAAAC3NxEyAAABrklEQVRoBa3BMQEAMBCEMPAv+iqA4Zcm8ol8Ip/IJ/KJfCKfyCfyiXwin8gn8ol8Ip/IJ/KJfCKfyCfyiXwin0iMk4TEOElIjJOExDhJSIyThMQ4SUiMk4TEOElIjJOExDhJSIyThMQ4SUiMk4TEOElIjJOExDhJSIyThMQ4SUiMk4TEOElIjJOExDhJSIyThMQ4SUiMk4TEOElIjJOExDhJSIyThMQ4SUiMk4TEOElIjJOExDhJSIyThMQ4SUiMk4TEOElIjJOExDhJSIyThMQ4SUiMk4TEOElIjJOExDhJSIyThMQ4SUiMk4TEOElIjJOExDhJSIyThMQ4SUiMk4TEOElIjJOExDhJSIyThMQ4SUiMk4TEOElIjJOExDhJSIyThMQ4SUiMk4TEOElIjJOExDhJSIyThMQ4SUiMk4TEOElIjJOExDhJSIyThMQ4SUiMk4TEOElIjJOExDhJSIyThMQ4SUiMk4TEOElIjJOExDhJSIyThMQ4Scgn8ol8Ip/IJ/KJfCKfyCfyiXwin8gn8ol8Ip/IJ/KJfCKfyCfyiXwin8gn8ol8Ip/IJw8mtFqPcukQSQAAAABJRU5ErkJggg=='),
(14, 3, 'R02', 366, 43, 70, 141, 1, 0, 'iVBORw0KGgoAAAANSUhEUgAAAEYAAACNCAAAAAAvalMvAAABzUlEQVRoBa3BsQ0AMAzDMOn/o93VgJcMJeUL+UK+kC/kC/lCvpAv5Av5Qr6QL+QL+UK+kC/kC/lCvpAv5Av5Qr6QL+QL+UJGuJAmI1xIkxEupMkIF9JkhAtpMsKFNBnhQpqMcCFNRriQJiNcSJMRLqTJCBfSZIQLaTLChTQZ4UKajHAhTUa4kCYjXEiTES6kyQgX0mSEC2kywoU0GeFCmoxwIU1GuJAmI1xIkxEupMkIF9JkhAtpMsKFNBnhQpqMcCFNRriQJiNcSJMRLqTJCBfSZIQLaTLChTQZ4UKajHAhTUa4kCYjXEiTES6kyQgX0mSEC2kywoU0GeFCmoxwIU1GuJAmI1xIkxEupMkIF9JkhAtpMsKFNBnhQpqMcCFNRriQJiNcSJMRLqTJCBfSZIQLaTLChTQZ4UKajHAhTUa4kCYjXEiTES6kyQgX0mSEC2kywoU0GeFCmoxwIU1GuJAmI1xIkxEupMkIF9JkhAtpMsKFNBnhQpqMcCFNRriQJiNcSJMRLqTJCBfSZIQLaTLChTQZ4UKajHAhTUa4kCZfyBfyhXwhX8gX8oV8IV/IF/KFfCFfyBfyhXwhX8gX8oV8IV/IF/KFfCFfyBcPR1VZjm8gP2kAAAAASUVORK5CYII='),
(15, 3, 'R03', 457, 44, 83, 139, 1, 0, 'iVBORw0KGgoAAAANSUhEUgAAAFMAAACLCAAAAAA4tPqeAAABy0lEQVRoBa3BMQEAMBCEMPAv+mqA4Ycm8p/8J//Jf/Kf/Cf/yX/yn/wn/8l/8p/8J//Jf/Kf/Cf/yX/yn/wn/8l/ksaNBEnjRoKkcSNB0riRIGncSJA0biRIGjcSJI0bCZLGjQRJ40aCpHEjQdK4kSBp3EiQNG4kSBo3EiSNGwmSxo0ESeNGgqRxI0HSuJEgadxIkDRuJEgaNxIkjRsJksaNBEnjRoKkcSNB0riRIGncSJA0biRIGjcSJI0bCZLGjQRJ40aCpHEjQdK4kSBp3EiQNG4kSBo3EiSNGwmSxo0ESeNGgqRxI0HSuJEgadxIkDRuJEgaNxIkjRsJksaNBEnjRoKkcSNB0riRIGncSJA0biRIGjcSJI0bCZLGjQRJ40aCpHEjQdK4kSBp3EiQNG4kSBo3EiSNGwmSxo0ESeNGgqRxI0HSuJEgadxIkDRuJEgaNxIkjRsJksaNBEnjRoKkcSNB0riRIGncSJA0biRIGjcSJI0bCZLGjQRJ40aCpHEjQdK4kSBp3EiQNG4kSBo3EiSNGwmSxo0ESeNGgvwn/8l/8p/8J//Jf/Kf/Cf/yX/yn/wn/8l/8p/8J//Jf/Kf/Cf/yX/yn/z3APBuWoxZFhx+AAAAAElFTkSuQmCC'),
(16, 3, 'R04', 569, 45, 79, 134, 1, 0, 'iVBORw0KGgoAAAANSUhEUgAAAE8AAACGCAAAAAC5uIgmAAABxklEQVRoBa3BMQEAMBCEMPAv+mqA4Ycm8pf8JX/JX/KX/CV/yV/yl/wlf8lf8pf8JX/JX/KX/CV/yV/yl4RxIUHCuJAgYVxIkDAuJEgYFxIkjAsJEsaFBAnjQoKEcSFBwriQIGFcSJAwLiRIGBcSJIwLCRLGhQQJ40KChHEhQcK4kCBhXEiQMC4kSBgXEiSMCwkSxoUECeNCgoRxIUHCuJAgYVxIkDAuJEgYFxIkjAsJEsaFBAnjQoKEcSFBwriQIGFcSJAwLiRIGBcSJIwLCRLGhQQJ40KChHEhQcK4kCBhXEiQMC4kSBgXEiSMCwkSxoUECeNCgoRxIUHCuJAgYVxIkDAuJEgYFxIkjAsJEsaFBAnjQoKEcSFBwriQIGFcSJAwLiRIGBcSJIwLCRLGhQQJ40KChHEhQcK4kCBhXEiQMC4kSBgXEiSMCwkSxoUECeNCgoRxIUHCuJAgYVxIkDAuJEgYFxIkjAsJEsaFBAnjQoKEcSFBwriQIGFcSJAwLiRIGBcSJIwLCRLGhQQJ40KChHEhQcK4kCBhXEiQMC4kyF/yl/wlf8lf8pf8JX/JX/KX/CV/yV/yl/wlf8lf8pf8JX/JXw+ssFyH35F2jQAAAABJRU5ErkJggg=='),
(17, 3, 'R05', 694, 43, 100, 147, 1, 0, 'iVBORw0KGgoAAAANSUhEUgAAAGQAAACTCAAAAABdHuMpAAAB3UlEQVRoBb3BQREAMBCEMPAveuuAuVcT+UA+kA/kA/lAPpAP5AP5QD6QD+QD+UA+kA/kA/lAPpAP5AP5QD6QD+QDSeNCmqRxIU3SuJAmaVxIkzQupEkaF9IkjQtpksaFNEnjQpqkcSFN0riQJmlcSJM0LqRJGhfSJI0LaZLGhTRJ40KapHEhTdK4kCZpXEiTNC6kSRoX0iSNC2mSxoU0SeNCmqRxIU3SuJAmaVxIkzQupEkaF9IkjQtpksaFNEnjQpqkcSFN0riQJmlcSJM0LqRJGhfSJI0LaZLGhTRJ40KapHEhTdK4kCZpXEiTNC6kSRoX0iSNC2mSxoU0SeNCmqRxIU3SuJAmaVxIkzQupEkaF9IkjQtpksaFNEnjQpqkcSFN0riQJmlcSJM0LqRJGhfSJI0LaZLGhTRJ40KapHEhTdK4kCZpXEiTNC6kSRoX0iSNC2mSxoU0SeNCmqRxIU3SuJAmaVxIkzQupEkaF9IkjQtpksaFNEnjQpqkcSFN0riQJmlcSJM0LqRJGhfSJI0LaZLGhTRJ40KapHEhTdK4kCZpXEiTD+QD+UA+kA/kA/lAPpAP5AP5QD6QD+QD+UA+kA/kA/lAPpAP5AP5QD6QD+QD+UA+kA/kA/ngAXM6XJTFXWWZAAAAAElFTkSuQmCC'),
(18, 3, 'R06', 807, 43, 76, 133, 1, 0, 'iVBORw0KGgoAAAANSUhEUgAAAEwAAACFCAAAAADUG0GLAAABvklEQVRoBa3BoQEAMAzDMPv/ozOekIJJ8pF8JB/JR/KRfCQfyUfykXwkH8lH8pF8JB/JR/KRfCQfyUfykXwkH8lH8pGUcCNLSriRJSXcyJISbmRJCTeypIQbWVLCjSwp4UaWlHAjS0q4kSUl3MiSEm5kSQk3sqSEG1lSwo0sKeFGlpRwI0tKuJElJdzIkhJuZEkJN7KkhBtZUsKNLCnhRpaUcCNLSriRJSXcyJISbmRJCTeypIQbWVLCjSwp4UaWlHAjS0q4kSUl3MiSEm5kSQk3sqSEG1lSwo0sKeFGlpRwI0tKuJElJdzIkhJuZEkJN7KkhBtZUsKNLCnhRpaUcCNLSriRJSXcyJISbmRJCTeypIQbWVLCjSwp4UaWlHAjS0q4kSUl3MiSEm5kSQk3sqSEG1lSwo0sKeFGlpRwI0tKuJElJdzIkhJuZEkJN7KkhBtZUsKNLCnhRpaUcCNLSriRJSXcyJISbmRJCTeypIQbWVLCjSwp4UaWlHAjS0q4kSUl3MiSEm5kSQk3sqSEG1lSwo0sKeFGlpRwI0s+ko/kI/lIPpKP5CP5SD6Sj+Qj+Ug+ko/kI/lIPpKP5KMHTgZZhim1iiMAAAAASUVORK5CYII='),
(19, 3, 'R07', 304, 224, 79, 139, 1, 0, 'iVBORw0KGgoAAAANSUhEUgAAAE8AAACLCAAAAAAFJpv4AAAB1klEQVRoBa3BsQEAMAzCMPv/o+kBMGSoJH/JX/KX/CV/yV/yl/wlf8lf8pf8JX/JX/KX/CV/yV/yl/wlLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggLZzIIC2cyCAtnMggf8lf8pf8JX/JX/KX/CV/yV/yl/wlf8lf8pf8JX/JX/KX/CV/yV/yl/wlf8lf8pf8JX89wWZajF6t14kAAAAASUVORK5CYII='),
(20, 3, 'R08', 567, 224, 82, 136, 1, 0, 'iVBORw0KGgoAAAANSUhEUgAAAFIAAACICAAAAABR4uMOAAAByklEQVRoBa3BMQEAMBCEMPAv+mqA4Ycm8p18J9/Jd/KdfCffyXfynXwn38l38p18J9/Jd/KdfCffyXeSxoUUSeNCiqRxIUXSuJAiaVxIkTQupEgaF1IkjQspksaFFEnjQoqkcSFF0riQImlcSJE0LqRIGhdSJI0LKZLGhRRJ40KKpHEhRdK4kCJpXEiRNC6kSBoXUiSNCymSxoUUSeNCiqRxIUXSuJAiaVxIkTQupEgaF1IkjQspksaFFEnjQoqkcSFF0riQImlcSJE0LqRIGhdSJI0LKZLGhRRJ40KKpHEhRdK4kCJpXEiRNC6kSBoXUiSNCymSxoUUSeNCiqRxIUXSuJAiaVxIkTQupEgaF1IkjQspksaFFEnjQoqkcSFF0riQImlcSJE0LqRIGhdSJI0LKZLGhRRJ40KKpHEhRdK4kCJpXEiRNC6kSBoXUiSNCymSxoUUSeNCiqRxIUXSuJAiaVxIkTQupEgaF1IkjQspksaFFEnjQoqkcSFF0riQImlcSJE0LqRIGhdSJI0LKZLGhRRJ40KKpHEhRdK4kCLfyXfynXwn38l38p18J9/Jd/KdfCffyXfynXwn38l38p18J9/Jd/KdfPcABFFciUfnDu0AAAAASUVORK5CYII='),
(21, 3, 'R09', 829, 222, 86, 142, 1, 0, NULL),
(22, 3, 'C01', 293, 102, 55, 80, 2, 0, 'iVBORw0KGgoAAAANSUhEUgAAADcAAABQCAAAAACYQbJ4AAABA0lEQVRYCZ3BoQEAMAzDMPv/ozMUEFIwSf7IH/kjf+SP/JE/8kf+yB/5I3/kj4xwkpIRTlIywklKRjhJyQgnKRnhJCUjnKRkhJOUjHCSkhFOUjLCSUpGOEnJCCcpGeEkJSOcpGSEk5SMcJKSEU5SMsJJSkY4SckIJykZ4SQlI5ykZISTlIxwkpIRTlIywklKRjhJyQgnKRnhJCUjnKRkhJOUjHCSkhFOUjLCSUpGOEnJCCcpGeEkJSOcpGSEk5SMcJKSEU5SMsJJSkY4SckIJykZ4SQlI5ykZISTlIxwkpIRTlIywklKRjhJyR/5I3/kj/yRP/JH/sgf+SN/5I/8kT/y5wHvjTRRiBkhLwAAAABJRU5ErkJggg=='),
(23, 3, 'C02', 178, 222, 86, 70, 2, 90, 'iVBORw0KGgoAAAANSUhEUgAAAFYAAABGCAAAAABz6zywAAAA2ElEQVRYCbXBsQ0AAAjDsOT/o2Gu2JBqS4VUSIVUSIVUSIVUSIVUSIVUSIVUSIVUSIVUSIVUSIVUSBge5JAwPMghYXiQQ8LwIIeE4UEOCcODHBKGBzkkDA9ySBge5JAwPMghYXiQQ8LwIIeE4UEOCcODHBKGBzkkDA9ySBge5JAwPMghYXiQQ8LwIIeE4UEOCcODHBKGBzkkDA9ySBge5JAwPMghYXiQQ8LwIIeE4UEOCcODHBKGBzkkDA9ySBge5JAKqZAKqZAKqZAKqZAKqZAKqZAKqZCKBWOQIUd9GNcmAAAAAElFTkSuQmCC'),
(24, 3, 'C03', 406, 231, 105, 58, 2, 90, 'iVBORw0KGgoAAAANSUhEUgAAAGkAAAA6CAAAAACW8yi/AAAAz0lEQVRYCb3BsQ3AQBDDMGn/oZ0FDAOX4kl5RV6RV+QVeUVekVfkFXlFXpFX5BV5RV6RJtzJJk24k02acCebNOFONmnCnWzShDvZpAl3skkT7mSTJtzJJk24k02acCebNOFONmnCnWzShDvZpAl3skkT7mSTJtzJJk24k02acCebNOFONmnCnWzShDvZpAl3skkT7mSTJtzJJk24k02a8INM0oQfZJIm/CCTdOFMJunCmUzShTOZpAtnMskr8oq8Iq/IK/KKvCKvyCvyirzyAaYpITupTPCeAAAAAElFTkSuQmCC'),
(25, 3, 'C04', 659, 233, 103, 52, 2, 90, 'iVBORw0KGgoAAAANSUhEUgAAAGcAAAA0CAAAAACyMHl8AAAAw0lEQVRYCb3BwQ0AMBDCsGT/oekCCOk+teUP+UP+kD/kD/lD/pA/5A/5Q/6QIZxJJ0M4k06GcCadDOFMOhnCmXQyhDPpZAhn0skQzqSTIZxJJ0M4k06GcCadDOFMOhnCmXQyhDPpZAhn0skQzqSTIZxJJ0M4k06GcCadDOFMOhnCmXQyhDPpZAhn0skQzqSTIZxJJ0M4k06GcCadDOFMOhnCmXQyhDPpZAhn0skQzqSTIZxJJ3/IH/KH/CF/yB/yh/zxALzgITUUDofjAAAAAElFTkSuQmCC'),
(26, 4, 'R01', 201, 45, 94, 135, 1, 0, 'iVBORw0KGgoAAAANSUhEUgAAAF4AAACHCAAAAAC6iLFVAAAB3UlEQVRoBbXBwREAMAzCMHv/oekEcPlUkq/kK/lKvpKv5Cv5Sr6Sr+Qr+Uq+kq/kK/lKvpKv5Cv5Sr6Sr+QrWcKFVLKEC6lkCRdSyRIupJIlXEglS7iQSpZwIZUs4UIqWcKFVLKEC6lkCRdSyRIupJIlXEglS7iQSpZwIZUs4UIqWcKFVLKEC6lkCRdSyRIupJIlXEglS7iQSpZwIZUs4UIqWcKFVLKEC6lkCRdSyRIupJIlXEglS7iQSpZwIZUs4UIqWcKFVLKEC6lkCRdSyRIupJIlXEglS7iQSpZwIZUs4UIqWcKFVLKEC6lkCRdSyRIupJIlXEglS7iQSpZwIZUs4UIqWcKFVLKEC6lkCRdSyRIupJIlXEglS7iQSpZwIZUs4UIqWcKFVLKEC6lkCRdSyRIupJIlXEglS7iQSpZwIZUs4UIqWcKFVLKEC6lkCRdSyRIupJIlXEglS7iQSpZwIZUs4UIqWcKFVLKEC6lkCRdSyRIupJIlXEglS7iQSpZwIZUs4UIqWcKFVLKEC6lkCRdSyRIupJIlXEglS7iQSpZwIZUs4UIqWcKFVLKEC6lkCRdSyVfylXwlX8lX8pV8JV/JV/KVfCVfyVfylXwlX8lX8pV8JV/JV/LVA6PwW4h7FSxKAAAAAElFTkSuQmCC'),
(27, 4, 'C02', 393, 218, 118, 75, 3, 90, 'iVBORw0KGgoAAAANSUhEUgAAAHYAAABLCAAAAACAKCy+AAAA5ElEQVRoBcXBQREAAAjDsNa/aHAwjtcSqZAKqZAKqZAKqZAKqZAKqZAKqZAKqZAKqZAKqZAKqZAKqZAKqZAKqZDb8CaZ3IY3yeQ2vEkmt+FNMrkNb5LJbXiTTG7Dm2RyG94kk9vwJpnchjfJ5Da8SSa34U0yuQ1vkslteJNMbsObZHIb3iST2/AmmdyGN8nkNrxJJrfhTTK5DW+SyW14k0xuw5tkchveJJPb8CaZ3IY3yeQ2vEkmt+FNMrkNb5LJbXiTTG7Dm2RyG94kkwqpkAqpkAqpkAqpkAqpkAqpkAqpkAqpWFWnIEwhdSNFAAAAAElFTkSuQmCC'),
(28, 4, 'SOT01', 441, 299, 110, 103, 4, 90, 'iVBORw0KGgoAAAANSUhEUgAAAG4AAABnCAAAAADEPyjvAAABdElEQVRoBb3BQQEAMAyEMPAvupNwvJbIV/KVfCVfyVfylXwlX8lX8pV8JV/JV1IcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDFIcjQxSHI0MUhyNDPKVfCVfyVfylXwlX8lX8pV8JV/JV/KVfCVfPf8cS2jEoE1wAAAAAElFTkSuQmCC');

-- --------------------------------------------------------

--
-- Estrutura para tabela `inspections`
--

CREATE TABLE `inspections` (
  `id` int(11) NOT NULL,
  `product_id` int(11) NOT NULL,
  `result` enum('OK','FAIL','IN_PROGRESS') DEFAULT 'IN_PROGRESS',
  `timestamp` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Despejando dados para a tabela `inspections`
--

INSERT INTO `inspections` (`id`, `product_id`, `result`, `timestamp`) VALUES
(1, 1, 'OK', '2025-10-24 17:58:04'),
(2, 2, 'FAIL', '2025-10-24 18:13:48'),
(3, 3, 'FAIL', '2025-10-24 18:18:20'),
(4, 3, 'FAIL', '2025-10-24 18:33:35'),
(5, 4, 'FAIL', '2025-10-24 18:45:15'),
(6, 4, 'FAIL', '2025-10-24 18:46:15'),
(7, 4, 'FAIL', '2025-10-25 00:46:37'),
(8, 2, 'FAIL', '2025-10-25 00:49:47');

-- --------------------------------------------------------

--
-- Estrutura para tabela `inspection_feedback`
--

CREATE TABLE `inspection_feedback` (
  `id` int(11) NOT NULL,
  `component_name` varchar(255) NOT NULL,
  `feedback` enum('GOOD','FAIL') NOT NULL,
  `timestamp` timestamp NOT NULL DEFAULT current_timestamp(),
  `user_id` int(11) DEFAULT NULL,
  `inspection_id` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Despejando dados para a tabela `inspection_feedback`
--

INSERT INTO `inspection_feedback` (`id`, `component_name`, `feedback`, `timestamp`, `user_id`, `inspection_id`) VALUES
(1, 'REP01', 'GOOD', '2025-10-24 17:58:25', 1, 1),
(2, 'COMP2', 'GOOD', '2025-10-24 17:58:25', 1, 1),
(3, 'C01', 'GOOD', '2025-10-25 00:50:54', 1, 8),
(4, 'COMP2', 'GOOD', '2025-10-25 00:50:54', 1, 8),
(5, 'COMP3', 'GOOD', '2025-10-25 00:50:54', 1, 8),
(6, 'COMP6', 'GOOD', '2025-10-25 00:50:54', 1, 8),
(7, 'COMP5', 'GOOD', '2025-10-25 00:50:54', 1, 8),
(8, 'COMP4', 'GOOD', '2025-10-25 00:50:54', 1, 8),
(9, 'COMP7', 'GOOD', '2025-10-25 00:50:54', 1, 8),
(10, 'COMP8', 'GOOD', '2025-10-25 00:50:54', 1, 8),
(11, 'COMP9', 'GOOD', '2025-10-25 00:50:54', 1, 8),
(12, 'COMP10', 'GOOD', '2025-10-25 00:50:54', 1, 8);

-- --------------------------------------------------------

--
-- Estrutura para tabela `inspection_results`
--

CREATE TABLE `inspection_results` (
  `id` int(11) NOT NULL,
  `inspection_id` int(11) NOT NULL,
  `component_id` int(11) NOT NULL,
  `cv_status` enum('OK','FAIL','UNKNOWN') NOT NULL,
  `ai_status` enum('OK','FAIL','UNKNOWN') NOT NULL,
  `ai_status_prob` float DEFAULT NULL,
  `cv_details` longtext DEFAULT NULL CHECK (json_valid(`cv_details`)),
  `final_status` enum('OK','FAIL') NOT NULL,
  `golden_roi_image` varchar(255) DEFAULT NULL,
  `produced_roi_image` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Despejando dados para a tabela `inspection_results`
--

INSERT INTO `inspection_results` (`id`, `inspection_id`, `component_id`, `cv_status`, `ai_status`, `ai_status_prob`, `cv_details`, `final_status`, `golden_roi_image`, `produced_roi_image`) VALUES
(1, 1, 1, 'OK', 'OK', 0.506051, '{\"message\": \"OK\", \"correlation_score\": 0.9357995986938477, \"ssim\": 1.0}', 'OK', 'iVBORw0KGgoAAAANSUhEUgAAAF0AAACSCAIAAACorVH1AAAgAElEQVR4AbTBedDn92HQ9/f78/09z+5qJa1WliVZki3nInjlQ3ISjkLiBALGZkoYEtKQIJUznZQkhMNOcGwnDoXpJAQHksIUOh2wnDJ0GCa4A0MMdXy3lDQ2kQ/5iB1LimX51LEr7e7z+37e/T4r5aAd+K+vly/92A8ChVETRcDIpJBNA4IgLZfCpBjJxDEZTqBMAnIQTHVFNpN', 'iVBORw0KGgoAAAANSUhEUgAAAF0AAACSCAIAAACorVH1AAAgAElEQVR4AbTBedDn92HQ9/f78/09z+5qJa1WliVZki3nInjlQ3ISjkLiBALGZkoYEtKQIJUznZQkhMNOcGwnDoXpJAQHksIUOh2wnDJ0GCa4A0MMdXy3lDQ2kQ/5iB1LimX51LEr7e7z+37e/T4r5aAd+K+vly/92A8ChVETRcDIpJBNA4IgLZfCpBjJxDEZTqBMAnIQTHVFNpN'),
(2, 1, 2, 'OK', 'OK', 0.505968, '{\"message\": \"OK\", \"correlation_score\": 0.9271470904350281, \"ssim\": 0.9271976196093528}', 'OK', 'iVBORw0KGgoAAAANSUhEUgAAAFkAAACVCAIAAAC8Q8E3AAAgAElEQVR4AbTBe9Dvd0HY+ff7832ec8tJgELI1QTUHQu5l6KSqNXOro6tuohUrUsSuupOrdqdKURdZ9zqtjuzrVYutqKArCbQum7tjqy7s1UJ4IVAhWIRTEIAK5zcgJjbSXLOeZ7f573f35MEaLXTv/p6efWdrwaJLxBCIqsKUAPZKjS2JJQZSqWyUmOWBKihUBONhs4UrGSlBKY', 'iVBORw0KGgoAAAANSUhEUgAAAFkAAACVCAIAAAC8Q8E3AAAgAElEQVR4AbTBe9Dvd0HY+ff7832ec8tJgELI1QTUHQu5l6KSqNXOro6tuohUrUsSuupOrdqdKURdZ9zqtjuzrVYutqKArCbQum7tjqy7s1UJ4IVAhWIRTEIAK5zcgJjbSXLOeZ7f573f35MEaLXTv/p6efWdrwaJLxBCIqsKUAPZKjS2JJQZSqWyUmOWBKihUBONhs4UrGSlBKY'),
(3, 2, 4, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_2_comp_4_g_b1274759.png', 'static/images\\results\\insp_2_comp_4_p_7119e108.png'),
(4, 2, 5, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_2_comp_5_g_9c3513fd.png', 'static/images\\results\\insp_2_comp_5_p_a7f3cd6f.png'),
(5, 2, 6, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_2_comp_6_g_b7301b8b.png', 'static/images\\results\\insp_2_comp_6_p_ace174a3.png'),
(6, 2, 7, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_2_comp_7_g_da9abaf6.png', 'static/images\\results\\insp_2_comp_7_p_e8956022.png'),
(7, 2, 8, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_2_comp_8_g_9d953af1.png', 'static/images\\results\\insp_2_comp_8_p_3820bd39.png'),
(8, 2, 9, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_2_comp_9_g_0b9dad47.png', 'static/images\\results\\insp_2_comp_9_p_358d0fb5.png'),
(9, 2, 10, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_2_comp_10_g_806e3c56.png', 'static/images\\results\\insp_2_comp_10_p_074732b4.png'),
(10, 2, 11, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_2_comp_11_g_02cd5b15.png', 'static/images\\results\\insp_2_comp_11_p_47dc857b.png'),
(11, 2, 12, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_2_comp_12_g_507957c8.png', 'static/images\\results\\insp_2_comp_12_p_f4a5b889.png'),
(12, 2, 3, 'OK', 'OK', 0.502902, '{\"message\": \"OK\", \"correlation_score\": 0.9492084383964539, \"ssim\": 1.0, \"color_similarity\": 1.0}', 'OK', 'static/images\\results\\insp_2_comp_3_g_00882855.png', 'static/images\\results\\insp_2_comp_3_p_9d5b763d.png'),
(13, 3, 13, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_3_comp_13_g_d47001f6.png', 'static/images\\results\\insp_3_comp_13_p_81034111.png'),
(14, 3, 14, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_3_comp_14_g_31268f81.png', 'static/images\\results\\insp_3_comp_14_p_3b68aafc.png'),
(15, 3, 15, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_3_comp_15_g_7ae2d97c.png', 'static/images\\results\\insp_3_comp_15_p_4d4632b3.png'),
(16, 3, 16, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_3_comp_16_g_3b0d4a94.png', 'static/images\\results\\insp_3_comp_16_p_1b446b1f.png'),
(17, 3, 17, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_3_comp_17_g_e21ad2b5.png', 'static/images\\results\\insp_3_comp_17_p_5123cc79.png'),
(18, 3, 18, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_3_comp_18_g_494249dc.png', 'static/images\\results\\insp_3_comp_18_p_b7bee2ae.png'),
(19, 3, 19, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_3_comp_19_g_972b4bc3.png', 'static/images\\results\\insp_3_comp_19_p_851cc8c9.png'),
(20, 3, 20, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_3_comp_20_g_623d152d.png', 'static/images\\results\\insp_3_comp_20_p_28656a5c.png'),
(21, 3, 21, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_3_comp_21_g_a44e555c.png', 'static/images\\results\\insp_3_comp_21_p_b0c265d5.png'),
(22, 3, 22, 'OK', 'OK', 0.502297, '{\"message\": \"OK\", \"correlation_score\": 0.9669350981712341, \"ssim\": 1.0, \"color_similarity\": 1.0}', 'OK', 'static/images\\results\\insp_3_comp_22_g_c1d08021.png', 'static/images\\results\\insp_3_comp_22_p_1ab194f6.png'),
(23, 3, 23, 'FAIL', 'OK', 0.504446, '{\"message\": \"Baixa Similaridade (SSIM: 0.36 < 0.6)\", \"correlation_score\": 0.7800709009170532, \"ssim\": 0.3598492009666261}', 'FAIL', 'static/images\\results\\insp_3_comp_23_g_51802f5b.png', 'static/images\\results\\insp_3_comp_23_p_09ef5c0a.png'),
(24, 3, 24, 'FAIL', 'OK', 0.503392, '{\"message\": \"Baixa Similaridade (SSIM: 0.35 < 0.6)\", \"correlation_score\": 0.7296034097671509, \"ssim\": 0.3543673860883965}', 'FAIL', 'static/images\\results\\insp_3_comp_24_g_e7fa5f19.png', 'static/images\\results\\insp_3_comp_24_p_eda7a117.png'),
(25, 3, 25, 'FAIL', 'OK', 0.502802, '{\"message\": \"Baixa Similaridade (SSIM: 0.46 < 0.6)\", \"correlation_score\": 0.7066466212272644, \"ssim\": 0.4608757750503582}', 'FAIL', 'static/images\\results\\insp_3_comp_25_g_4ecdcba1.png', 'static/images\\results\\insp_3_comp_25_p_587b10ce.png'),
(26, 4, 13, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_4_comp_13_g_49df4cea.png', 'static/images\\results\\insp_4_comp_13_p_ef0affbf.png'),
(27, 4, 14, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_4_comp_14_g_f2033e9c.png', 'static/images\\results\\insp_4_comp_14_p_7387e3c3.png'),
(28, 4, 15, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_4_comp_15_g_6cbaf65b.png', 'static/images\\results\\insp_4_comp_15_p_7a47bc9a.png'),
(29, 4, 16, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_4_comp_16_g_7e5d6e09.png', 'static/images\\results\\insp_4_comp_16_p_e3cf10a6.png'),
(30, 4, 17, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_4_comp_17_g_273aa932.png', 'static/images\\results\\insp_4_comp_17_p_66373ac5.png'),
(31, 4, 18, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_4_comp_18_g_087f1555.png', 'static/images\\results\\insp_4_comp_18_p_9dd27d80.png'),
(32, 4, 19, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_4_comp_19_g_78afc795.png', 'static/images\\results\\insp_4_comp_19_p_4e186134.png'),
(33, 4, 20, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_4_comp_20_g_e96c713a.png', 'static/images\\results\\insp_4_comp_20_p_eff1786e.png'),
(34, 4, 21, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_4_comp_21_g_437065bd.png', 'static/images\\results\\insp_4_comp_21_p_8bafeb33.png'),
(35, 4, 22, 'OK', 'OK', 0.502297, '{\"message\": \"OK\", \"correlation_score\": 0.9669350981712341, \"ssim\": 1.0, \"color_similarity\": 1.0}', 'OK', 'static/images\\results\\insp_4_comp_22_g_65635e15.png', 'static/images\\results\\insp_4_comp_22_p_084af55b.png'),
(36, 4, 23, 'FAIL', 'OK', 0.504446, '{\"message\": \"Baixa Similaridade (SSIM: 0.36 < 0.6)\", \"correlation_score\": 0.7800709009170532, \"ssim\": 0.3598492009666261}', 'FAIL', 'static/images\\results\\insp_4_comp_23_g_3b74a9ac.png', 'static/images\\results\\insp_4_comp_23_p_70e6aa96.png'),
(37, 4, 24, 'FAIL', 'OK', 0.503392, '{\"message\": \"Baixa Similaridade (SSIM: 0.35 < 0.6)\", \"correlation_score\": 0.7296034097671509, \"ssim\": 0.3543673860883965}', 'FAIL', 'static/images\\results\\insp_4_comp_24_g_d4821a97.png', 'static/images\\results\\insp_4_comp_24_p_45e889e4.png'),
(38, 4, 25, 'FAIL', 'OK', 0.502802, '{\"message\": \"Baixa Similaridade (SSIM: 0.46 < 0.6)\", \"correlation_score\": 0.7066466212272644, \"ssim\": 0.4608757750503582}', 'FAIL', 'static/images\\results\\insp_4_comp_25_g_5beaec34.png', 'static/images\\results\\insp_4_comp_25_p_8675a91f.png'),
(39, 5, 26, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_5_comp_26_g_916cc5da.png', 'static/images\\results\\insp_5_comp_26_p_8947fcc7.png'),
(40, 5, 27, 'OK', 'FAIL', 0.489898, '{\"message\": \"OK\", \"correlation_score\": 0.9232529401779175, \"ssim\": 1.0, \"color_similarity\": 1.0}', 'FAIL', 'static/images\\results\\insp_5_comp_27_g_011cf971.png', 'static/images\\results\\insp_5_comp_27_p_718c76f2.png'),
(41, 5, 28, 'FAIL', 'FAIL', 0.492521, '{\"message\": \"Rota\\u00e7\\u00e3o Incorreta (Esperado: 90\\u00b0, Encontrado: 0\\u00b0)\", \"correlation_score\": 0.854256808757782}', 'FAIL', 'static/images\\results\\insp_5_comp_28_g_f6e4b37a.png', 'static/images\\results\\insp_5_comp_28_p_f54ba692.png'),
(42, 6, 26, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_6_comp_26_g_bc883543.png', 'static/images\\results\\insp_6_comp_26_p_8593b29d.png'),
(43, 6, 27, 'FAIL', 'FAIL', 0.489819, '{\"message\": \"Falha na Cor (Similaridade: 0.38 < 0.7)\", \"correlation_score\": 0.9027736186981201, \"ssim\": 0.9232557732297595, \"color_similarity\": 0.3837780994388011}', 'FAIL', 'static/images\\results\\insp_6_comp_27_g_05e9f2b4.png', 'static/images\\results\\insp_6_comp_27_p_11c8f322.png'),
(44, 6, 28, 'FAIL', 'FAIL', 0.492239, '{\"message\": \"Rota\\u00e7\\u00e3o Incorreta (Esperado: 90\\u00b0, Encontrado: 0\\u00b0)\", \"correlation_score\": 0.829706072807312}', 'FAIL', 'static/images\\results\\insp_6_comp_28_g_8d0d4da2.png', 'static/images\\results\\insp_6_comp_28_p_4a789c0e.png'),
(45, 7, 26, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_7_comp_26_g_5e6ed48d.png', 'static/images\\results\\insp_7_comp_26_p_d7866ce4.png'),
(46, 7, 27, 'OK', 'OK', 0.500108, '{\"message\": \"OK\", \"correlation_score\": 0.9232529401779175, \"ssim\": 1.0, \"color_similarity\": 1.0}', 'OK', 'static/images\\results\\insp_7_comp_27_g_a8a255ca.png', 'static/images\\results\\insp_7_comp_27_p_aff25c8e.png'),
(47, 7, 28, 'FAIL', 'FAIL', 0.498917, '{\"message\": \"Rota\\u00e7\\u00e3o Incorreta (Esperado: 90\\u00b0, Encontrado: 0\\u00b0)\", \"correlation_score\": 0.854256808757782}', 'FAIL', 'static/images\\results\\insp_7_comp_28_g_b267e7f6.png', 'static/images\\results\\insp_7_comp_28_p_7ca4e6b5.png'),
(48, 8, 3, 'OK', 'FAIL', 0.498043, '{\"message\": \"OK\", \"correlation_score\": 0.9492084383964539, \"ssim\": 1.0, \"color_similarity\": 1.0}', 'FAIL', 'static/images\\results\\insp_8_comp_3_g_5e356e91.png', 'static/images\\results\\insp_8_comp_3_p_f1851b55.png'),
(49, 8, 4, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_8_comp_4_g_bb00b826.png', 'static/images\\results\\insp_8_comp_4_p_e8cad374.png'),
(50, 8, 5, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_8_comp_5_g_d81b86c2.png', 'static/images\\results\\insp_8_comp_5_p_16899c9f.png'),
(51, 8, 6, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_8_comp_6_g_4e64240a.png', 'static/images\\results\\insp_8_comp_6_p_27f391cf.png'),
(52, 8, 7, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_8_comp_7_g_e2171cb7.png', 'static/images\\results\\insp_8_comp_7_p_c0930ca4.png'),
(53, 8, 8, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_8_comp_8_g_bd5f0642.png', 'static/images\\results\\insp_8_comp_8_p_2b0dec62.png'),
(54, 8, 9, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_8_comp_9_g_aff981ae.png', 'static/images\\results\\insp_8_comp_9_p_639db234.png'),
(55, 8, 10, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_8_comp_10_g_f736333e.png', 'static/images\\results\\insp_8_comp_10_p_8af60c01.png'),
(56, 8, 11, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_8_comp_11_g_9c68b716.png', 'static/images\\results\\insp_8_comp_11_p_372d8ffb.png'),
(57, 8, 12, 'FAIL', 'UNKNOWN', 0, '{\"message\": \"Arquivo de template do pacote n\\u00e3o encontrado.\"}', 'FAIL', 'static/images\\results\\insp_8_comp_12_g_7af19a8d.png', 'static/images\\results\\insp_8_comp_12_p_ede52e68.png');

-- --------------------------------------------------------

--
-- Estrutura para tabela `packages`
--

CREATE TABLE `packages` (
  `id` int(11) NOT NULL,
  `name` varchar(255) NOT NULL,
  `body_matrix` varchar(255) DEFAULT NULL,
  `body_mask` varchar(255) DEFAULT NULL,
  `presence_threshold` float DEFAULT 0.35,
  `ssim_threshold` float DEFAULT 0.6
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Despejando dados para a tabela `packages`
--

INSERT INTO `packages` (`id`, `name`, `body_matrix`, `body_mask`, `presence_threshold`, `ssim_threshold`) VALUES
(1, 'R0204', 'iVBORw0KGgoAAAANSUhEUgAAACQAAABcCAIAAACJE4BjAAAXTElEQVRoBbXBB5ieZYHu8f/9vN83NYWE9Gkh9JBGU3FFVBTS6EWaiFhWLIiKlJAGiqy0FJoiywFBXRUREFTQhCQQFBESCARIISSZJEDaTGYm85X3fe79JhTDHj2713Wu/f104L+flcWYZqltjCSDbISMMRUyBoERNggQFYJodpF4h8BgkACFJJfL54Kk8Q9d3Ke6thzLJhKNQRh', NULL, 0.35, 0.6),
(2, 'C0805', 'static/images\\packages\\pkg_2_C0805_fc09052f.png', NULL, 0.35, 0.6),
(3, 'CAPACITOR', 'static/images\\packages\\pkg_3_CAPACITOR_a8e081e8.png', NULL, 0.35, 0.6),
(4, 'SOT23', 'static/images\\packages\\pkg_4_SOT23_1c666482.png', NULL, 0.35, 0.6),
(5, 'REP0204', 'static/images\\packages\\pkg_5_REP0204_template_6419e8f9.png', 'static/images\\packages\\pkg_5_REP0204_mask_23d6190a.png', 0.35, 0.6);

-- --------------------------------------------------------

--
-- Estrutura para tabela `products`
--

CREATE TABLE `products` (
  `id` int(11) NOT NULL,
  `name` varchar(255) NOT NULL,
  `golden_image` varchar(255) NOT NULL,
  `fiducials` longtext NOT NULL CHECK (json_valid(`fiducials`))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Despejando dados para a tabela `products`
--

INSERT INTO `products` (`id`, `name`, `golden_image`, `fiducials`) VALUES
(1, 'Primeiro', 'static/uploads\\golden_a44d34c7-5d81-4e2b-8c1b-5a9c0a00d60e_4fiduciais.png', '[{\"name\":\"FID1\",\"x\":73,\"y\":516,\"r\":22},{\"name\":\"FID2\",\"x\":77,\"y\":48,\"r\":22},{\"name\":\"FID3\",\"x\":928,\"y\":48,\"r\":21},{\"name\":\"FID4\",\"x\":924,\"y\":516,\"r\":22}]'),
(2, 'Segundo', 'static/uploads\\golden_0a9141af-fd4b-43b8-8f8f-a0c879b2fc39_4fiduciais.png', '[{\"name\":\"FID1\",\"x\":73,\"y\":516,\"r\":22},{\"name\":\"FID2\",\"x\":924,\"y\":516,\"r\":22},{\"name\":\"FID3\",\"x\":928,\"y\":48,\"r\":21},{\"name\":\"FID4\",\"x\":77,\"y\":48,\"r\":22}]'),
(3, 'Terceiro', 'static/uploads\\golden_1e8c2997-4213-4afe-9bab-2dab11d7839e_4fiduciais.png', '[{\"name\":\"FID1\",\"x\":73,\"y\":516,\"r\":22},{\"name\":\"FID2\",\"x\":924,\"y\":516,\"r\":22},{\"name\":\"FID3\",\"x\":928,\"y\":48,\"r\":21},{\"name\":\"FID4\",\"x\":77,\"y\":48,\"r\":22}]'),
(4, '01', 'static/uploads\\golden_77043f23-8026-4854-b709-f78ea701e1d6_4fiduciais.png', '[{\"name\":\"FID1\",\"x\":73,\"y\":516,\"r\":22},{\"name\":\"FID2\",\"x\":924,\"y\":516,\"r\":22},{\"name\":\"FID3\",\"x\":928,\"y\":48,\"r\":21},{\"name\":\"FID4\",\"x\":77,\"y\":48,\"r\":22}]');

-- --------------------------------------------------------

--
-- Estrutura para tabela `training_samples`
--

CREATE TABLE `training_samples` (
  `id` int(11) NOT NULL,
  `product_id` int(11) NOT NULL,
  `component_id` int(11) NOT NULL,
  `golden_path` varchar(255) DEFAULT NULL,
  `produced_path` varchar(255) DEFAULT NULL,
  `label` enum('GOOD','FAIL') NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Despejando dados para a tabela `training_samples`
--

INSERT INTO `training_samples` (`id`, `product_id`, `component_id`, `golden_path`, `produced_path`, `label`, `created_at`) VALUES
(1, 1, 1, 'iVBORw0KGgoAAAANSUhEUgAAAF0AAACSCAIAAACorVH1AAAgAElEQVR4AbTBedDn92HQ9/f78/09z+5qJa1WliVZki3nInjlQ3ISjkLiBALGZkoYEtKQIJUznZQkhMNOcGwnDoXpJAQHksIUOh2wnDJ0GCa4A0MMdXy3lDQ2kQ/5iB1LimX51LEr7e7z+37e/T4r5aAd+K+vly/92A8ChVETRcDIpJBNA4IgLZfCpBjJxDEZTqBMAnIQTHVFNpN', 'iVBORw0KGgoAAAANSUhEUgAAAF0AAACSCAIAAACorVH1AAAgAElEQVR4AbTBedDn92HQ9/f78/09z+5qJa1WliVZki3nInjlQ3ISjkLiBALGZkoYEtKQIJUznZQkhMNOcGwnDoXpJAQHksIUOh2wnDJ0GCa4A0MMdXy3lDQ2kQ/5iB1LimX51LEr7e7z+37e/T4r5aAd+K+vly/92A8ChVETRcDIpJBNA4IgLZfCpBjJxDEZTqBMAnIQTHVFNpN', 'GOOD', '2025-10-24 17:58:25'),
(2, 1, 2, 'iVBORw0KGgoAAAANSUhEUgAAAFkAAACVCAIAAAC8Q8E3AAAgAElEQVR4AbTBe9Dvd0HY+ff7832ec8tJgELI1QTUHQu5l6KSqNXOro6tuohUrUsSuupOrdqdKURdZ9zqtjuzrVYutqKArCbQum7tjqy7s1UJ4IVAhWIRTEIAK5zcgJjbSXLOeZ7f573f35MEaLXTv/p6efWdrwaJLxBCIqsKUAPZKjS2JJQZSqWyUmOWBKihUBONhs4UrGSlBKY', 'iVBORw0KGgoAAAANSUhEUgAAAFkAAACVCAIAAAC8Q8E3AAAgAElEQVR4AbTBe9Dvd0HY+ff7832ec8tJgELI1QTUHQu5l6KSqNXOro6tuohUrUsSuupOrdqdKURdZ9zqtjuzrVYutqKArCbQum7tjqy7s1UJ4IVAhWIRTEIAK5zcgJjbSXLOeZ7f573f35MEaLXTv/p6efWdrwaJLxBCIqsKUAPZKjS2JJQZSqWyUmOWBKihUBONhs4UrGSlBKY', 'GOOD', '2025-10-24 17:58:25'),
(3, 2, 3, 'static/images\\results\\insp_8_comp_3_g_5e356e91.png', 'static/images\\results\\insp_8_comp_3_p_f1851b55.png', 'GOOD', '2025-10-25 00:50:54'),
(4, 2, 4, 'static/images\\results\\insp_8_comp_4_g_bb00b826.png', 'static/images\\results\\insp_8_comp_4_p_e8cad374.png', 'GOOD', '2025-10-25 00:50:54'),
(5, 2, 5, 'static/images\\results\\insp_8_comp_5_g_d81b86c2.png', 'static/images\\results\\insp_8_comp_5_p_16899c9f.png', 'GOOD', '2025-10-25 00:50:54'),
(6, 2, 8, 'static/images\\results\\insp_8_comp_8_g_bd5f0642.png', 'static/images\\results\\insp_8_comp_8_p_2b0dec62.png', 'GOOD', '2025-10-25 00:50:54'),
(7, 2, 7, 'static/images\\results\\insp_8_comp_7_g_e2171cb7.png', 'static/images\\results\\insp_8_comp_7_p_c0930ca4.png', 'GOOD', '2025-10-25 00:50:54'),
(8, 2, 6, 'static/images\\results\\insp_8_comp_6_g_4e64240a.png', 'static/images\\results\\insp_8_comp_6_p_27f391cf.png', 'GOOD', '2025-10-25 00:50:54'),
(9, 2, 9, 'static/images\\results\\insp_8_comp_9_g_aff981ae.png', 'static/images\\results\\insp_8_comp_9_p_639db234.png', 'GOOD', '2025-10-25 00:50:54'),
(10, 2, 10, 'static/images\\results\\insp_8_comp_10_g_f736333e.png', 'static/images\\results\\insp_8_comp_10_p_8af60c01.png', 'GOOD', '2025-10-25 00:50:54'),
(11, 2, 11, 'static/images\\results\\insp_8_comp_11_g_9c68b716.png', 'static/images\\results\\insp_8_comp_11_p_372d8ffb.png', 'GOOD', '2025-10-25 00:50:54'),
(12, 2, 12, 'static/images\\results\\insp_8_comp_12_g_7af19a8d.png', 'static/images\\results\\insp_8_comp_12_p_ede52e68.png', 'GOOD', '2025-10-25 00:50:54');

-- --------------------------------------------------------

--
-- Estrutura para tabela `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL,
  `username` varchar(255) NOT NULL,
  `password_hash` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Despejando dados para a tabela `users`
--

INSERT INTO `users` (`id`, `username`, `password_hash`) VALUES
(1, 'admin', 'scrypt:32768:8:1$4cmmhlUVFXTSWYyZ$4339f4bb37f5745a761a175d326fa3155b495ae29019303a9caa0ff186474597ef7d84b6e6deb6d33bb72b2b0c663a61315901a2c942761b6afd20fde7132826');

--
-- Índices para tabelas despejadas
--

--
-- Índices de tabela `components`
--
ALTER TABLE `components`
  ADD PRIMARY KEY (`id`),
  ADD KEY `product_id` (`product_id`),
  ADD KEY `fk_components_package` (`package_id`);

--
-- Índices de tabela `inspections`
--
ALTER TABLE `inspections`
  ADD PRIMARY KEY (`id`),
  ADD KEY `product_id` (`product_id`);

--
-- Índices de tabela `inspection_feedback`
--
ALTER TABLE `inspection_feedback`
  ADD PRIMARY KEY (`id`),
  ADD KEY `inspection_id` (`inspection_id`),
  ADD KEY `user_id` (`user_id`);

--
-- Índices de tabela `inspection_results`
--
ALTER TABLE `inspection_results`
  ADD PRIMARY KEY (`id`),
  ADD KEY `inspection_id` (`inspection_id`);

--
-- Índices de tabela `packages`
--
ALTER TABLE `packages`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `name` (`name`);

--
-- Índices de tabela `products`
--
ALTER TABLE `products`
  ADD PRIMARY KEY (`id`);

--
-- Índices de tabela `training_samples`
--
ALTER TABLE `training_samples`
  ADD PRIMARY KEY (`id`),
  ADD KEY `product_id` (`product_id`),
  ADD KEY `component_id` (`component_id`);

--
-- Índices de tabela `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `username` (`username`);

--
-- AUTO_INCREMENT para tabelas despejadas
--

--
-- AUTO_INCREMENT de tabela `components`
--
ALTER TABLE `components`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=38;

--
-- AUTO_INCREMENT de tabela `inspections`
--
ALTER TABLE `inspections`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=10;

--
-- AUTO_INCREMENT de tabela `inspection_feedback`
--
ALTER TABLE `inspection_feedback`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=13;

--
-- AUTO_INCREMENT de tabela `inspection_results`
--
ALTER TABLE `inspection_results`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=58;

--
-- AUTO_INCREMENT de tabela `packages`
--
ALTER TABLE `packages`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- AUTO_INCREMENT de tabela `products`
--
ALTER TABLE `products`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- AUTO_INCREMENT de tabela `training_samples`
--
ALTER TABLE `training_samples`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=13;

--
-- AUTO_INCREMENT de tabela `users`
--
ALTER TABLE `users`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- Restrições para tabelas despejadas
--

--
-- Restrições para tabelas `components`
--
ALTER TABLE `components`
  ADD CONSTRAINT `components_ibfk_1` FOREIGN KEY (`product_id`) REFERENCES `products` (`id`) ON DELETE CASCADE,
  ADD CONSTRAINT `components_ibfk_2` FOREIGN KEY (`package_id`) REFERENCES `packages` (`id`) ON DELETE SET NULL;

--
-- Restrições para tabelas `inspections`
--
ALTER TABLE `inspections`
  ADD CONSTRAINT `inspections_ibfk_1` FOREIGN KEY (`product_id`) REFERENCES `products` (`id`) ON DELETE CASCADE;

--
-- Restrições para tabelas `inspection_feedback`
--
ALTER TABLE `inspection_feedback`
  ADD CONSTRAINT `inspection_feedback_ibfk_1` FOREIGN KEY (`inspection_id`) REFERENCES `inspections` (`id`) ON DELETE CASCADE,
  ADD CONSTRAINT `inspection_feedback_ibfk_2` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE SET NULL;

--
-- Restrições para tabelas `inspection_results`
--
ALTER TABLE `inspection_results`
  ADD CONSTRAINT `inspection_results_ibfk_1` FOREIGN KEY (`inspection_id`) REFERENCES `inspections` (`id`) ON DELETE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
