-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Oct 04, 2025 at 12:25 AM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.0.30

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `smt_inspection_new`
--

-- --------------------------------------------------------

--
-- Table structure for table `components`
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
-- Dumping data for table `components`
--

INSERT INTO `components` (`id`, `product_id`, `name`, `x`, `y`, `width`, `height`, `package_id`, `rotation`, `inspection_mask`) VALUES
(1, 1, 'COMP1', 215, 50, 60, 118, 1, 0, NULL),
(2, 1, 'COMP2', 370, 48, 67, 125, 1, 0, NULL),
(3, 2, 'COMP1', 207, 53, 78, 124, 1, 0, NULL),
(4, 2, 'COMP2', 570, 221, 70, 139, 1, 0, NULL),
(5, 2, 'COMP3', 717, 414, 170, 108, 1, 0, NULL),
(6, 4, 'COMP1', 214, 50, 64, 121, 1, 0, NULL),
(7, 4, 'COMP2', 293, 95, 53, 83, 1, 0, NULL),
(8, 4, 'COMP3', 367, 52, 68, 119, 1, 0, NULL),
(9, 4, 'COMP4', 463, 50, 69, 124, 1, 0, NULL),
(10, 4, 'COMP5', 575, 49, 66, 122, 1, 0, NULL),
(11, 4, 'COMP6', 702, 46, 76, 127, 1, 0, NULL),
(12, 4, 'COMP7', 808, 43, 67, 134, 1, 0, NULL),
(13, 4, 'COMP8', 166, 227, 103, 63, 1, 0, NULL),
(14, 4, 'COMP9', 198, 299, 96, 96, 1, 0, NULL),
(15, 4, 'COMP10', 307, 224, 67, 130, 1, 0, NULL),
(16, 4, 'COMP11', 410, 220, 84, 78, 1, 0, NULL),
(17, 4, 'COMP12', 449, 303, 95, 93, 1, 0, NULL),
(18, 4, 'COMP13', 570, 229, 75, 129, 1, 0, NULL),
(19, 4, 'COMP14', 663, 228, 98, 64, 1, 0, NULL),
(20, 4, 'COMP15', 708, 301, 98, 96, 1, 0, NULL),
(21, 4, 'COMP16', 834, 220, 78, 133, 1, 0, NULL),
(22, 4, 'COMP17', 193, 408, 160, 116, 1, 0, NULL),
(23, 4, 'COMP18', 462, 410, 158, 114, 1, 0, NULL),
(24, 4, 'COMP19', 727, 414, 155, 108, 1, 0, NULL),
(25, 5, 'COMP1', 204, 41, 83, 144, 25, 0, NULL),
(26, 5, 'COMP2', 360, 38, 82, 143, 25, 0, NULL),
(27, 5, 'COMP3', 708, 291, 109, 109, 27, 0, NULL),
(28, 5, 'COMP4', 168, 223, 97, 75, 28, 0, NULL),
(29, 6, 'COMP1', 206, 27, 80, 165, 1, 0, NULL),
(30, 6, 'COMP2', 403, 216, 109, 81, 1, 0, NULL),
(31, 6, 'COMP3', 455, 404, 179, 128, 1, 0, NULL),
(32, 7, 'COMP1', 180, 39, 118, 148, 1, 0, NULL),
(33, 7, 'COMP2', 290, 222, 97, 129, 1, 0, NULL),
(34, 7, 'COMP3', 441, 288, 104, 115, 1, 0, NULL),
(35, 7, 'COMP4', 183, 417, 183, 101, 1, 0, NULL),
(36, 8, 'COMP1', 292, 214, 101, 145, 1, 0, NULL),
(37, 8, 'COMP2', 438, 287, 117, 123, 1, 0, NULL),
(38, 9, 'COMP1', 209, 50, 77, 124, 38, 0, 'iVBORw0KGgoAAAANSUhEUgAAAE0AAAB8CAAAAAAJRGJkAAABpUlEQVRoBa3BQQ0AMBDDsIQ/6I5ApPvMlp/kJ/lJfpKf5Cf5SX6Sn+Qn+Ul+kp/kJ/lJfpKfJIybBAnjJkHCuEmQMG4SJIybBAnjJkHCuEmQMG4SJIybBAnjJkHCuEmQMG4SJIybBAnjJkHCuEmQMG4SJIybBAnjJkHCuEmQMG4SJIybBAnjJkHCuEmQMG4SJIybBAnjJkHCuEmQMG4SJIybBAnjJkHCuEmQMG4SJIybBAnjJkHCuEmQMG4SJIybBAnjJkHCuEmQMG4SJIybBAnjJkHCuEmQMG4SJIybBAnjJkHCuEmQMG4SJIybBAnjJkHCuEmQMG4SJIybBAnjJkHCuEmQMG4SJIybBAnjJkHCuEmQMG4SJIybBAnjJkHCuEmQMG4SJIybBAnjJkHCuEmQMG4SJIybBAnjJkHCuEmQMG4SJIybBAnjJkHCuEmQMG4SJIybBAnjJkHCuEmQMG4SJIybBAnjJkHCuEmQMG4SJIybBAnjJkHCuEmQMG4SJIybBAnjJkF+kp/kJ/lJfpKf5Cf5SX6Sn+Qn+Ul+kp/kJ/lJfnozAFp9C6vVGwAAAABJRU5ErkJggg=='),
(39, 9, 'COMP2', 364, 47, 73, 136, 38, 0, 'iVBORw0KGgoAAAANSUhEUgAAAEkAAACICAAAAACOrJkRAAAByElEQVRoBa3BsQ0AMAzDMOn/o93dMJClpPwiv8gv8ov8Ir/IL/KL/CK/yC/yi/wiv8gv8ov8Ir/ILzKEmxQZwk2KDOEmRYZwkyJDuEmRIdykyBBuUmQINykyhJsUGcJNigzhJkWGcJMiQ7hJkSHcpMgQblJkCDcpMoSbFBnCTYoM4SZFhnCTIkO4SZEh3KTIEG5SZAg3KTKEmxQZwk2KDOEmRYZwkyJDuEmRIdykyBBuUmQINykyhJsUGcJNigzhJkWGcJMiQ7hJkSHcpMgQblJkCDcpMoSbFBnCTYoM4SZFhnCTIkO4SZEh3KTIEG5SZAg3KTKEmxQZwk2KDOEmRYZwkyJDuEmRIdykyBBuUmQINykyhJsUGcJNigzhJkWGcJMiQ7hJkSHcpMgQblJkCDcpMoSbFBnCTYoM4SZFhnCTIkO4SZEh3KTIEG5SZAg3KTKEmxQZwk2KDOEmRYZwkyJDuEmRIdykyBBuUmQINykyhJsUGcJNigzhJkWGcJMiQ7hJkSHcpMgQblJkCDcpMoSbFBnCTYoM4SZFfpFf5Bf5RX6RX+QX+UV+kV/kF/lFfpFf5Bf5RX6RX+QX+UV+kV/kF/lFfpFfHoeTW4n7NWNgAAAAAElFTkSuQmCC'),
(40, 9, 'COMP3', 186, 413, 186, 119, 40, 90, 'iVBORw0KGgoAAAANSUhEUgAAALoAAAB3CAAAAACREwVHAAABv0lEQVR4AdXBgQnDAAzEQGn/ob8TmCch1PhOzpKz5Cw5S86Ss+QsOUvOkrPkLDlLzpKz5Cw5S86Ss+S58Cl5R54Ln5J35LnwKXlHngufknfkufApeUeeC5+Sd+S58Cl5R54Ln5J35LnwB9LIc+EPpJFZWCSNzMIiaWQWFkkjs7BIGpmFRdLILCySRmZhkTQyC4ukkVlYJI3MwiJpZBYWSSOzsEgamYVF0sgsLJJGZmGRNDILi6SRWVgkjczCImlkFhZJI7OwSBqZhUXSyCwskkZmYZE0MguLpJFZWCSNzMIiaWQWFkkjs7BIGpmFRdLILCySRmZhkTQyC4ukkVlYJI3MwiJpZBYWSSOzsEgamYVF0sgsLJJGZmGRNDILi6SRWVgkjczCImlkFhZJI7OwSBqZhUXSyCwskkZmYZE0MguLpJFZWCSNzMIiaWQWFkkjs7BIGpmFRdLIc+EPpJHnwh9II8+FP5BGngufknfkufApeUeeC5+Sd+S58Cl5R54Ln5J35LnwKXlHngufknfkLDlLzpKz5Cw5S86Ss+QsOUvOkrPkLDlLzpKz5Cw5S86Ss+QsOUvOkrPkLDlLzpKzfpi5SXgkjBgJAAAAAElFTkSuQmCC'),
(41, 13, 'COMP1', 201, 51, 90, 141, 38, 0, NULL),
(42, 14, 'COMP1', 200, 41, 93, 148, 38, 0, 'iVBORw0KGgoAAAANSUhEUgAAAF0AAACUCAAAAADU/XpjAAABzklEQVRoBbXBQQEAMAyEMPAv+uaAvpbIT/KT/CQ/yU/yk/wkP8lP8pP8JD/JT/KT/CQ/yU/yk/wkP8lP8pP8JD/JT/KTlHGSIGWcJEgZJwlSxkmClHGSIGWcJEgZJwlSxkmClHGSIGWcJEgZJwlSxkmClHGSIGWcJEgZJwlSxkmClHGSIGWcJEgZJwlSxkmClHGSIGWcJEgZJwlSxkmClHGSIGWcJEgZJwlSxkmClHGSIGWcJEgZJwlSxkmClHGSIGWcJEgZJwlSxkmClHGSIGWcJEgZJwlSxkmClHGSIGWcJEgZJwlSxkmClHGSIGWcJEgZJwlSxkmClHGSIGWcJEgZJwlSxkmClHGSIGWcJEgZJwlSxkmClHGSIGWcJEgZJwlSxkmClHGSIGWcJEgZJwlSxkmClHGSIGWcJEgZJwlSxkmClHGSIGWcJEgZJwlSxkmClHGSIGWcJEgZJwlSxkmClHGSIGWcJEgZJwlSxkmClHGSIGWcJEgZJwlSxkmClHGSIGWcJEgZJwlSxkmClHGSIGWcJMhP8pP8JD/JT/KT/CQ/yU/yk/wkP8lP8pP8JD/JT/KT/CQ/yU/yk/wkP8lP8pP8JD/JT/KT/CQ/PScPWpWKdtkDAAAAAElFTkSuQmCC'),
(43, 14, 'COMP2', 352, 36, 93, 147, 38, 0, 'iVBORw0KGgoAAAANSUhEUgAAAF0AAACTCAAAAADJ+ErbAAAB90lEQVRoBbXBwREAMAzCMHv/oekEcPlUkp/kJ/lJfpKf5Cf5SX6Sn+Qn+Ul+kp/kJ/lJfpKf5Cf5SX6Sn+Qn+Ul+kp/kJ/lJfpKf5CfZwoEUsoUDKWQLB1LIFg6kkC0cSCFbOJBCtnAghWzhQArZwoEUsoUDKWQLB1LIFg6kkC0cSCFbOJBCtnAghWzhQArZwoEUsoUDKWQLB1LIFg6kkC0cSCFbOJBCtnAghWzhQArZwoEUsoUDKWQLB1LIFg6kkC0cSCFbOJBCtnAghWzhQArZwoEUsoUDKWQLB1LIFg6kkC0cSCFbOJBCtnAghWzhQArZwoEUsoUDKWQLB1LIFg6kkC0cSCFbOJBCtnAghWzhQArZwoEUsoUDKWQLB1LIFg6kkC0cSCFbOJBCtnAghWzhQArZwoEUsoUDKWQLB1LIFg6kkC0cSCFbOJBCtnAghWzhQArZwoEUsoUDKWQLB1LIFg6kkC0cSCFbOJBCtnAghWzhQArZwoEUsoUDKWQLB1LIFg6kkC0cSCFbOJBCtnAghWzhQArZwoEUsoUDKWQLB1LIFg6kkC0cSCFbOJBCtnAghWzhQArZwoEUsoUDKWQLB1LIFg6kkC0cSCE/yU/yk/wkP8lP8pP8JD/JT/KT/CQ/yU/yk/wkP8lP8pP8JD/JT/KT/PQAOyBdlBbsk58AAAAASUVORK5CYII='),
(44, 14, 'COMP3', 453, 49, 81, 131, 38, 0, 'iVBORw0KGgoAAAANSUhEUgAAAFEAAACDCAAAAADQEqjOAAAByUlEQVRoBa3BsQkAMBADsbv9h3Z6g+GLSPKb/Ca/yW/ym/wmv8lv8pv8Jr/Jb/Kb/Ca/yW/ym/wmSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiSziRIks4kSJLOJEiv8lv8pv8Jr/Jb/Kb/Ca/yW/ym/wmv8lv8pv8Jr/Jb/Kb/Ca/yW/y2wP3dFqELrde8wAAAABJRU5ErkJggg=='),
(45, 14, 'COMP4', 558, 46, 100, 139, 38, 0, 'iVBORw0KGgoAAAANSUhEUgAAAGQAAACLCAAAAACym2PfAAAB4UlEQVRoBb3BQQEAMBDCsNa/aOYA7rVEPpAP5AP5QD6QD+QD+UA+kA/kA/lAPpAP5AP5QD6QD+QD+UA+kCFcSCNDuJBGhnAhjQzhQhoZwoU0MoQLaWQIF9LIEC6kkSFcSCNDuJBGhnAhjQzhQhoZwoU0MoQLaWQIF9LIEC6kkSFcSCNDuJBGhnAhjQzhQhoZwoU0MoQLaWQIF9LIEC6kkSFcSCNDuJBGhnAhjQzhQhoZwoU0MoQLaWQIF9LIEC6kkSFcSCNDuJBGhnAhjQzhQhoZwoU0MoQLaWQIF9LIEC6kkSFcSCNDuJBGhnAhjQzhQhoZwoU0MoQLaWQIF9LIEC6kkSFcSCNDuJBGhnAhjQzhQhoZwoU0MoQLaWQIF9LIEC6kkSFcSCNDuJBGhnAhjQzhQhoZwoU0MoQLaWQIF9LIEC6kkSFcSCNDuJBGhnAhjQzhQhoZwoU0MoQLaWQIF9LIEC6kkSFcSCNDuJBGhnAhjQzhQhoZwoU0MoQLaWQIF9LIEC6kkSFcSCNDuJBGhnAhjQzhQhoZwoU0MoQLaWQIF9LIEC6kkSFcSCNDuJBGPpAP5AP5QD6QD+QD+UA+kA/kA/lAPpAP5AP5QD6QD+QD+UA+kA/kA/lAPpAP5AP54AH6YFqM1UQE+wAAAABJRU5ErkJggg=='),
(46, 14, 'COMP5', 700, 44, 87, 148, 38, 0, 'iVBORw0KGgoAAAANSUhEUgAAAFcAAACUCAAAAADD3+qqAAAB70lEQVRoBbXBwQ0AMBDCsGT/oekCIN2ntvwhf8gf8of8IX/IH/KH/CF/yB/yh/whf8gf8of8IX/IH/KH/CF/yB8yhBPpZAgn0skQTqSTIZxIJ0M4kU6GcCKdDOFEOhnCiXQyhBPpZAgn0skQTqSTIZxIJ0M4kU6GcCKdDOFEOhnCiXQyhBPpZAgn0skQTqSTIZxIJ0M4kU6GcCKdDOFEOhnCiXQyhBPpZAgn0skQTqSTIZxIJ0M4kU6GcCKdDOFEOhnCiXQyhBPpZAgn0skQTqSTIZxIJ0M4kU6GcCKdDOFEOhnCiXQyhBPpZAgn0skQTqSTIZxIJ0M4kU6GcCKdDOFEOhnCiXQyhBPpZAgn0skQTqSTIZxIJ0M4kU6GcCKdDOFEOhnCiXQyhBPpZAgn0skQTqSTIZxIJ0M4kU6GcCKdDOFEOhnCiXQyhBPpZAgn0skQTqSTIZxIJ0M4kU6GcCKdDOFEOhnCiXQyhBPpZAgn0skQTqSTIZxIJ0M4kU6GcCKdDOFEOhnCiXQyhBPpZAgn0skQTqSTIZxIJ0M4kU6GcCKdDOFEOhnCiXQyhBPp5A/5Q/6QP+QP+UP+kD/kD/lD/pA/5A/5Q/6QP+QP+UP+kD/kD/lD/pA/5A/5Q/6QP+QP+UP+kD/kD/lD/pA/HuwNWZVNa6yuAAAAAElFTkSuQmCC'),
(47, 14, 'COMP6', 796, 43, 90, 140, 38, 0, 'iVBORw0KGgoAAAANSUhEUgAAAFoAAACMCAAAAADZpOHsAAAB6ElEQVRoBbXBwQ0AMBDCsGT/oekCCN2ntnwj38g38o18I9/IN/KNfCPfyDfyjXwj38g38o18I9/IN/KNfCPfyBIOpJMlHEgnSziQTpZwIJ0s4UA6WcKBdLKEA+lkCQfSyRIOpJMlHEgnSziQTpZwIJ0s4UA6WcKBdLKEA+lkCQfSyRIOpJMlHEgnSziQTpZwIJ0s4UA6WcKBdLKEA+lkCQfSyRIOpJMlHEgnSziQTpZwIJ0s4UA6WcKBdLKEA+lkCQfSyRIOpJMlHEgnSziQTpZwIJ0s4UA6WcKBdLKEA+lkCQfSyRIOpJMlHEgnSziQTpZwIJ0s4UA6WcKBdLKEA+lkCQfSyRIOpJMlHEgnSziQTpZwIJ0s4UA6WcKBdLKEA+lkCQfSyRIOpJMlHEgnSziQTpZwIJ0s4UA6WcKBdLKEA+lkCQfSyRIOpJMlHEgnSziQTpZwIJ0s4UA6WcKBdLKEA+lkCQfSyRIOpJMlHEgnSziQTpZwIJ0s4UA6WcKBdLKEA+lkCQfSyRIOpJMlHEgnSziQTpZwIJ0s4UA6WcKBdLKEA+lkCQfSyRIOpJMlHEgnSziQTpZwIJ18I9/IN/KNfCPfyDfyjXwj38g38o18I9/IN/KNfCPfyDfyjXwj38g38o18I988lgxcjSoa8iMAAAAASUVORK5CYII='),
(48, 14, 'COMP7', 301, 225, 83, 123, 38, 0, 'iVBORw0KGgoAAAANSUhEUgAAAFMAAAB7CAAAAAAtJuOHAAAByElEQVRoBa3BwQ0AMBDCsGT/oekCIN2jtvwn/8l/8p/8J//Jf/Kf/Cf/yX/yn/wn/8l/8p/8J//JfzKEA2lkCAfSyBAOpJEhHEgjQziQRoZwII0M4UAaGcKBNDKEA2lkCAfSyBAOpJEhHEgjQziQRoZwII0M4UAaGcKBNDKEA2lkCAfSyBAOpJEhHEgjQziQRoZwII0M4UAaGcKBNDKEA2lkCAfSyBAOpJEhHEgjQziQRoZwII0M4UAaGcKBNDKEA2lkCAfSyBAOpJEhHEgjQziQRoZwII0M4UAaGcKBNDKEA2lkCAfSyBAOpJEhHEgjQziQRoZwII0M4UAaGcKBNDKEA2lkCAfSyBAOpJEhHEgjQziQRoZwII0M4UAaGcKBNDKEA2lkCAfSyBAOpJEhHEgjQziQRoZwII0M4UAaGcKBNDKEA2lkCAfSyBAOpJEhHEgjQziQRoZwII0M4UAaGcKBNDKEA2lkCAfSyBAOpJEhHEgjQziQRoZwII0M4UAaGcKBNDKEA2lkCAfSyBAOpJEhHEgjQziQRoZwII0M4UAaGcKBNDKEA2lkCAfSyBAOpJH/5D/5T/6T/+Q/+U/+k//kP/lP/pP/HhnKW3y+nQQXAAAAAElFTkSuQmCC'),
(49, 14, 'COMP8', 562, 230, 96, 135, 38, 0, 'iVBORw0KGgoAAAANSUhEUgAAAGAAAACHCAAAAADMsgPeAAABwUlEQVRoBbXBQREAMBCEMPAveuuAuU8T+Uw+k8/kM/lMPpPP5DP5TD6Tz+Qz+Uw+k88kjANJEsaBJAnjQJKEcSBJwjiQJGEcSJIwDiRJGAeSJIwDSRLGgSQJ40CShHEgScI4kCRhHEiSMA4kSRgHkiSMA0kSxoEkCeNAkoRxIEnCOJAkYRxIkjAOJEkYB5IkjANJEsaBJAnjQJKEcSBJwjiQJGEcSJIwDiRJGAeSJIwDSRLGgSQJ40CShHEgScI4kCRhHEiSMA4kSRgHkiSMA0kSxoEkCeNAkoRxIEnCOJAkYRxIkjAOJEkYB5IkjANJEsaBJAnjQJKEcSBJwjiQJGEcSJIwDiRJGAeSJIwDSRLGgSQJ40CShHEgScI4kCRhHEiSMA4kSRgHkiSMA0kSxoEkCeNAkoRxIEnCOJAkYRxIkjAOJEkYB5IkjANJEsaBJAnjQJKEcSBJwjiQJGEcSJIwDiRJGAeSJIwDSRLGgSQJ40CShHEgScI4kCRhHEiSMA4kSRgHkiSMA0nymXwmn8ln8pl8Jp/JZ/KZfCafyWfymXwmn8ln8pl8Jp/JZ/KZfCafyWfymXwmn8ln8pl8Jp89Kg9ZiE1nEMYAAAAASUVORK5CYII='),
(50, 14, 'COMP9', 823, 223, 102, 147, 38, 0, 'iVBORw0KGgoAAAANSUhEUgAAAGYAAACTCAAAAABZ6zMUAAAB6ElEQVRoBb3BQQEAMBDCsNa/aOYA7rVEvpAv5Av5Qr6QL+QL+UK+kC/kC/lCvpAv5Av5Qr6QL+QL+UK+kCpcyCJVuJBFqnAhi1ThQhapwoUsUoULWaQKF7JIFS5kkSpcyCJVuJBFqnAhi1ThQhapwoUsUoULWaQKF7JIFS5kkSpcyCJVuJBFqnAhi1ThQhapwoUsUoULWaQKF7JIFS5kkSpcyCJVuJBFqnAhi1ThQhapwoUsUoULWaQKF7JIFS5kkSpcyCJVuJBFqnAhi1ThQhapwoUsUoULWaQKF7JIFS5kkSpcyCJVuJBFqnAhi1ThQhapwoUsUoULWaQKF7JIFS5kkSpcyCJVuJBFqnAhi1ThQhapwoUsUoULWaQKF7JIFS5kkSpcyCJVuJBFqnAhi1ThQhapwoUsUoULWaQKF7JIFS5kkSpcyCJVuJBFqnAhi1ThQhapwoUsUoULWaQKF7JIFS5kkSpcyCJVuJBFqnAhi1ThQhapwoUsUoULWaQKF7JIFS5kkSpcyCJVuJBFqnAhi1ThQhapwoUsUoULWaQKF7LIF/KFfCFfyBfyhXwhX8gX8oV8IV/IF/KFfCFfyBfyhXwhX8gX8oV8IV/IF/KFfCFfyBfyhXwhX8gX8oV8IV/IF/KFfPEAu41XlOvivokAAAAASUVORK5CYII='),
(51, 14, 'COMP10', 295, 100, 52, 87, 54, 0, 'iVBORw0KGgoAAAANSUhEUgAAADQAAABXCAAAAABucznDAAABCUlEQVRYCZ3BsQ0AMAzDMOn/o93JgKcMJeWDfJAP8kE+yAf5IB/kg3yQD/JBPsgH+SAjXKRkhIuUjHCRkhEuUjLCRUpGuEjJCBcpGeEiJSNcpGSEi5SMcJGSES5SMsJFSka4SMkIFykZ4SIlI1ykZISLlIxwkZIRLlIywkVKRrhIyQgXKRnhIiUjXKRkhIuUjHCRkhEuUjLCRUpGuEjJCBcpGeEiJSNcpGSEi5SMcJGSES5SMsJFSka4SMkIFykZ4SIlI1ykZISLlIxwkZIRLlIywkVKRrhIyQgXKRnhIiUjXKTkg3yQD/JBPsgH+SAf5IN8kA/yQT7IB/kgH+SDfJAP8kE+yAf58AC0UjFYK8kagwAAAABJRU5ErkJggg=='),
(52, 14, 'COMP11', 170, 223, 96, 72, 54, 90, 'iVBORw0KGgoAAAANSUhEUgAAAGAAAABICAAAAAAsDK+/AAAA2klEQVRYCbXBMQEAAAjDsNa/aDAwDo4lUiZlUiZlUiZlUiZlUiZlUiZlUiZlUiZlUiZlUiZlUiZlchjeJJDD8CaBHIY3CeQwvEkgh+FNAjkMbxLIYXiTQA7DmwRyGN4kkMPwJoEchjcJ5DC8SSCH4U0COQxvEshheJNADsObBHIY3iSQw/AmgRyGNwnkMLxJIIfhTQI5DG8SyGF4k0AOw5sEchjeJJDD8CaBHIY3CeQwvEkgh+FNAjkMbxJImZRJmZRJmZRJmZRJmZRJmZRJmZRJmZRJmZRJmZQtq74eSdBHNusAAAAASUVORK5CYII='),
(53, 14, 'COMP12', 404, 230, 106, 58, 54, 90, 'iVBORw0KGgoAAAANSUhEUgAAAGoAAAA6CAAAAAB9xJO8AAAAzUlEQVRYCb3BsREAMBDCMHv/ockCHMUXkeQb+Ua+kW/kG/lGvpFv5Bv5Rr6Rb6QKBzJJFQ5kkiocyCRVOJBJqnAgk1ThQCapwoFMUoUDmaQKBzJJFQ5kkiocyCRVOJBJqnAgk1ThQCapwoFMUoUDmaQKBzJJFQ5kkiocyCRVOJBJqnAgk1ThQCapwoFMUoUDmaQKBzJJFQ5kkiocyCRVOJBJqnAgk1ThQCapwoFMUoUDmaQKBzLJN/KNfCPfyDfyjXwj38g38o18I9/INw/HwSE7MDi6xwAAAABJRU5ErkJggg=='),
(54, 14, 'COMP13', 667, 219, 84, 63, 54, 90, 'iVBORw0KGgoAAAANSUhEUgAAAFQAAAA/CAAAAABbM7CEAAAAyklEQVRYCbXBoQEAAAjDsPb/o4fEYCZI5IE8kAfyQB7IA3kgD+SBPJAH8kAeyAN5IA/kgTyQB/JAHsgDeSAP5IHcQkmW3EJJltxCSZbcQkmW3EJJltxCSZbcQkmW3EJJltxCSZbcQkmW3EJJltxCSZbcQkmW3EJJltxCSZbcQkmW3EJJltxCSZbcQkmW3EJJltxCSZbcQkmW3EJJltxCSZbcQkmW3EJJltxCSZbcQkmW3EJJltxCSZbcQkmW3EJJljyQB/JAHsgDeTDKHSBATg+j7gAAAABJRU5ErkJggg=='),
(55, 14, 'COMP14', 198, 299, 100, 100, 27, 90, 'iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAAAAABVicqIAAABdElEQVRoBb3BsRHAQAzDMHL/oZU6X1jnxoAckANyQA7IATkgB+SAHJADckAOSBE6mUkROplJETqZSRE6mUkROplJETqZSRE6mUkROplJETqZSRE6mUkROpnJI+zJTB5hT2byCHsyk0fYk5k8wp7M5BH2ZCaPsCczeYQ9mckj7MlMHmFPZvIIezKTR9iTmRShk5kUoZOZFKGTmRShk5kUoZOZFKGTmRShk5kUoZOZFKGTmRShk5kUoZOZFGFP/qQIe/InRdiTPynCnvxJEfbkT4qwJ39ShD35kyLsyZ8UYU/+pAh78idF6GQmRehkJkXoZCZF6GQmRehkJkXoZCZF6GQmRehkJkXoZCZF6GQmRehkJo+wJzN5hD2ZySPsyUweYU9m8gh7MpNH2JOZPMKezOQR9mQmj7AnM3mEPZnJI+zJTB5hT2ZShE5mUoROZlKETmZShE5mUoROZnJADsgBOSAH5IAckANyQA7IATkgB+SAHJADH8bzSGURSzKKAAAAAElFTkSuQmCC'),
(56, 14, 'COMP15', 445, 297, 110, 101, 27, 90, 'iVBORw0KGgoAAAANSUhEUgAAAG4AAABlCAAAAACJ94nkAAABiUlEQVRoBb3BwQ0DMQDDMGn/od0BUsC+PELKU/KUPCVPyVPylDwlT8lT8pQ8JU/JU/KUdKGTiXShk4l0oZOJdKGTiXShk4l0oZOJdKGTiXShk4l0oZOJnMJ3MpFT+E4mcgrfyURO4TuZyCl8JxM5he9kIqfwnUzkFL6TiZzCdzKRU/hOJnIK38lETuE7mcgpfCcTOYXvZCKn8J1MpAudTKQLnUykC51MpAudTKQLnUykC51MpAudTKQLnUykC51MpAs35A/pwg35Q7pwQ/6QLtyQP6QLN+QP6cIN+UO6cEP+kC7ckD+kCzfkD+nCDflDunBD/pAu3JA/pAudTKQLnUykC51MpAudTKQLnUykC51MpAudTOQQLshEDuGCTOQQLshEDuGCTOQQLshEDuGCTOQQLshEDuGCTOQQLshEDuGCTOQQLshEDuGCTOQQLshEDuGCTKQLnUykC51MpAudTKQLnUykC51MpAudTKQLnUzkKXlKnpKn5Cl5Sp6Sp+QpeUqekqfkKXnqBxXjSWZ6caUlAAAAAElFTkSuQmCC'),
(57, 14, 'COMP16', 708, 300, 103, 102, 27, 90, 'iVBORw0KGgoAAAANSUhEUgAAAGcAAABmCAAAAADzdtCAAAABiklEQVRoBb3BsREDMRDEMLL/otepA83e6QMB8oa8IW/IG/KGvCFvyBvyhrwhb8gb8oZshJk0shFm0shGmEkjG2EmjWyEmTSyEWbSyEaYSSMbYSaNnIVb0shZuCWNnIVb0shZuCWNnIVb0shZuCWNnIVb0shZuCWNnIVb0shZuCWNnIVb0shZuCWNbISZNLIRZtLIRphJIxthJo1shJk0shFm0shGmEkjG2EmjWyEmTSyEWbSyEaYSSMb4Z78k41wT/7JRrgn/2Qj3JN/shHuyT/ZCPfkn2yEe/JPNsI9+Scb4Z78k41wT/7JRphJIxthJo1shJk0shFm0shGmEkjG2EmjWyEmTSyEWbSyEaYSSMbYSaNbISZNHIQPpBGDsIH0shB+EAaOQgfSCMH4QNp5CB8II0chA+kkYPwgTRyED6QRg7CB9LIRphJIxthJo1shJk0shFm0shGmEkjG2EmjWyEmTSyEWbSyBvyhrwhb8gb8oa8IW/IG/KGvCFvyBvyhrwhb8gb8oa88QOUcUZn2txEyQAAAABJRU5ErkJggg=='),
(58, 14, 'COMP17', 189, 415, 177, 119, 61, 90, 'iVBORw0KGgoAAAANSUhEUgAAALEAAAB3CAAAAABp8/6wAAABtklEQVR4AdXBwQ3AQAzDMGn/od0J/AhwaBBSrpFr5Bq5Rq6Ra+QauUaukWvkGrlGrpFrZCQ8JzMyEp6TGRkJz8mMjITnZEZGwnMyIyPhOZmRkfCczMhIeE5mZCQ8JzMyEp6TGanCL2RGqrBGOqnCGumkCmukkyqskU6qsEY6qcIa6aQKa6STKqyRTqqwRjqpwhrppAprpJMqrJFOqrBGOqnCGumkCmukkyqskU6qsEY6qcIa6aQKa6STKqyRTqqwRjqpwhrppAprpJMqrJFOqrBGOqnCGumkCmukkyqskU6qsEY6qcIa6aQKa6STKqyRTqqwRjqpwhrppAprpJMqrJFOqrBGOqnCGumkCmukkyqskU6qsEY6qcIa6aQKa6STKqyRTqqwRjqpwhrppAprpJMqrJFOqrBGOqnCGumkCmukkyqskU6qsEY6qcIa6aQKv5AZqcIvZEZGwnMyIyPhOZmRkfCczMhIeE5mZCQ8JzMyEp6TGRkJz8mMjITnZEZGwnMyI9fINXKNXCPXyDVyjVwj18g1co1cI9fINXKNXCPXyDVyjVwj18g1co1cI9fINXKNXCPXfNuNTHgHK5MRAAAAAElFTkSuQmCC'),
(59, 14, 'COMP18', 438, 412, 201, 118, 61, 90, 'iVBORw0KGgoAAAANSUhEUgAAAMkAAAB2CAAAAABHMeh5AAABxUlEQVR4Ad3BgQ3DAAzDMOn/o70TjLQdAoSUK+QKuUKukCvkCrlCrpAr5Aq5Qq6QK+QKuUKukCvkCrlCrpDXwpfkIXktfEkektfCl+QheS18SR6S18KX5CF5LXxJHpLXwpfkIRkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SSgbBFKhkIW6SS18L/SSWvhf+TSl4LX5KH5LXwJXlIXgtfkofktfAleUheC1+Sh+S18CV5SF4LX5KH5LXwJXlIrpAr5Aq5Qq6QK+QKuUKukCvkCrlCrpAr5Aq5Qq6QK+QKuUKukCvkCrniB0F7SndpQ87bAAAAAElFTkSuQmCC'),
(60, 14, 'COMP19', 711, 415, 173, 110, 61, 90, 'iVBORw0KGgoAAAANSUhEUgAAAK0AAABuCAAAAABwuMyFAAABp0lEQVR4AdXBQQoDQQwEsar/P7pz9qlhWGIsySVyiVwil8glcolcIpfIJXKJXCKXyCVyiVwij8LnpJFH4XPSyKPwOWnkUficNPIofE4aeRQ+J408Cp+TRh6Fz0kjj8LnpJFH4XPSyKPwDzJIFfbIIFXYI4NUYY8MUoU9MkgV9sggVdgjg1RhjwxShT0ySBX2yCBV2CODVGGPDFKFPTJIFfbIIFXYI4NUYY8MUoU9MkgV9sggVdgjg1RhjwxShT0ySBX2yCBV2CODVGGPDFKFPTJIFfbIIFXYI4NUYY8MUoU9MkgV9sggVdgjg1RhjwxShT0ySBX2yCBV2CODVGGPDFKFPTJIFfbIIFXYI4NUYY8MUoU9MkgV9sggVdgjg1RhjwxShT0ySBX2yCBV2CODVGGPDFKFPTJIFfbIIFXYI4NU4U+kkSr8iTTyKHxOGnkUPieNPAqfk0Yehc9JI4/C56SRR+Fz0sij8Dlp5FH4nDTyKHxOGnkUPieNPAqfk0Yehc9JI5fIJXKJXCKXyCVyiVwil8glcolcIpfIJXKJXCKXyCVyiVzyA3YvS28E+KEPAAAAAElFTkSuQmCC');

-- --------------------------------------------------------

--
-- Table structure for table `inspections`
--

CREATE TABLE `inspections` (
  `id` int(11) NOT NULL,
  `product_id` int(11) NOT NULL,
  `produced_image_path` varchar(255) DEFAULT NULL,
  `overall_status` enum('OK','FAIL','IN_PROGRESS') DEFAULT 'IN_PROGRESS',
  `total_ok` int(11) DEFAULT 0,
  `total_fail` int(11) DEFAULT 0,
  `inspection_timestamp` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `inspections`
--

INSERT INTO `inspections` (`id`, `product_id`, `produced_image_path`, `overall_status`, `total_ok`, `total_fail`, `inspection_timestamp`) VALUES
(1, 14, 'static/uploads\\inspected_33412bd9-f465-476f-a1a9-307cbc9981e6_testeMissing.png', 'FAIL', 0, 19, '2025-10-03 19:52:59'),
(2, 14, 'static/uploads\\inspected_660933d2-a713-4881-8257-5ceee3c93c92_4fiduciais.png', 'FAIL', 0, 19, '2025-10-03 19:54:20'),
(3, 14, 'static/uploads\\inspected_a979d21a-e521-42b3-b191-39c2960be21c_4fiduciais.png', 'FAIL', 0, 19, '2025-10-03 20:02:58');

-- --------------------------------------------------------

--
-- Table structure for table `inspection_results`
--

CREATE TABLE `inspection_results` (
  `id` int(11) NOT NULL,
  `inspection_id` int(11) NOT NULL,
  `component_id` int(11) NOT NULL,
  `status` enum('OK','FAIL') NOT NULL,
  `rotation` varchar(10) DEFAULT NULL,
  `displacement_x` int(11) DEFAULT NULL,
  `displacement_y` int(11) DEFAULT NULL,
  `metrics` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL CHECK (json_valid(`metrics`)),
  `roi_golden_path` varchar(255) DEFAULT NULL,
  `roi_produced_path` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `inspection_results`
--

INSERT INTO `inspection_results` (`id`, `inspection_id`, `component_id`, `status`, `rotation`, `displacement_x`, `displacement_y`, `metrics`, `roi_golden_path`, `roi_produced_path`) VALUES
(1, 1, 51, 'FAIL', '180°', 0, 0, '{\"ssim\": 0.06444200948913838, \"match_val\": 0.05797313526272774, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_1_51.png', 'static/roi_images\\roi_produced_1_51.png'),
(2, 1, 52, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.20468850976534128, \"match_val\": 0.6504688262939453, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_1_52.png', 'static/roi_images\\roi_produced_1_52.png'),
(3, 1, 53, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.2726308038186756, \"match_val\": 0.6616566181182861, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_1_53.png', 'static/roi_images\\roi_produced_1_53.png'),
(4, 1, 54, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.26401171352377145, \"match_val\": 0.5840429067611694, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_1_54.png', 'static/roi_images\\roi_produced_1_54.png'),
(5, 1, 58, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.40282524926713736, \"match_val\": 0.7608340978622437, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_1_58.png', 'static/roi_images\\roi_produced_1_58.png'),
(6, 1, 59, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.3519654681201376, \"match_val\": 0.7502828240394592, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_1_59.png', 'static/roi_images\\roi_produced_1_59.png'),
(7, 1, 60, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.43986022888103554, \"match_val\": 0.7560718059539795, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_1_60.png', 'static/roi_images\\roi_produced_1_60.png'),
(8, 1, 42, 'FAIL', '180°', 0, 0, '{\"ssim\": 0.024957271045699416, \"match_val\": -0.013110264204442501, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_1_42.png', 'static/roi_images\\roi_produced_1_42.png'),
(9, 1, 43, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.19252222733032268, \"match_val\": 0.5419322848320007, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_1_43.png', 'static/roi_images\\roi_produced_1_43.png'),
(10, 1, 44, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.26651328458719986, \"match_val\": 0.5843580961227417, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_1_44.png', 'static/roi_images\\roi_produced_1_44.png'),
(11, 1, 45, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.21539855576175523, \"match_val\": 0.5499747395515442, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_1_45.png', 'static/roi_images\\roi_produced_1_45.png'),
(12, 1, 46, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.24542148103147343, \"match_val\": 0.5740088820457458, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_1_46.png', 'static/roi_images\\roi_produced_1_46.png'),
(13, 1, 47, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.24616150498405567, \"match_val\": 0.5690230131149292, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_1_47.png', 'static/roi_images\\roi_produced_1_47.png'),
(14, 1, 48, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.03462774081359769, \"match_val\": -0.1601853370666504, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_1_48.png', 'static/roi_images\\roi_produced_1_48.png'),
(15, 1, 49, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.2450947609748164, \"match_val\": 0.5646154880523682, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_1_49.png', 'static/roi_images\\roi_produced_1_49.png'),
(16, 1, 50, 'FAIL', '180°', 0, 0, '{\"ssim\": 0.017627219744946868, \"match_val\": -0.1187044009566307, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_1_50.png', 'static/roi_images\\roi_produced_1_50.png'),
(17, 1, 55, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.2881096546718529, \"match_val\": 0.47752565145492554, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_1_55.png', 'static/roi_images\\roi_produced_1_55.png'),
(18, 1, 56, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.277078461353143, \"match_val\": 0.45719319581985474, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_1_56.png', 'static/roi_images\\roi_produced_1_56.png'),
(19, 1, 57, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.26928625955150737, \"match_val\": 0.48355239629745483, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_1_57.png', 'static/roi_images\\roi_produced_1_57.png'),
(20, 2, 51, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.3315637733843529, \"match_val\": 0.6702873110771179, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_2_51.png', 'static/roi_images\\roi_produced_2_51.png'),
(21, 2, 52, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.22299658302198874, \"match_val\": 0.6583858132362366, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_2_52.png', 'static/roi_images\\roi_produced_2_52.png'),
(22, 2, 53, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.2863121400799731, \"match_val\": 0.6551619172096252, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_2_53.png', 'static/roi_images\\roi_produced_2_53.png'),
(23, 2, 54, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.28049549953773784, \"match_val\": 0.5852913856506348, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_2_54.png', 'static/roi_images\\roi_produced_2_54.png'),
(24, 2, 58, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.42385472651103545, \"match_val\": 0.7729659676551819, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_2_58.png', 'static/roi_images\\roi_produced_2_58.png'),
(25, 2, 59, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.3619640233608222, \"match_val\": 0.7466282844543457, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_2_59.png', 'static/roi_images\\roi_produced_2_59.png'),
(26, 2, 60, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.45782079341954857, \"match_val\": 0.7594281435012817, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_2_60.png', 'static/roi_images\\roi_produced_2_60.png'),
(27, 2, 42, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.23821836828964216, \"match_val\": 0.5878744721412659, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_2_42.png', 'static/roi_images\\roi_produced_2_42.png'),
(28, 2, 43, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.25628743457436876, \"match_val\": 0.5942277908325195, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_2_43.png', 'static/roi_images\\roi_produced_2_43.png'),
(29, 2, 44, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.3475131016379556, \"match_val\": 0.6280724406242371, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_2_44.png', 'static/roi_images\\roi_produced_2_44.png'),
(30, 2, 45, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.25543370514694363, \"match_val\": 0.5895667672157288, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_2_45.png', 'static/roi_images\\roi_produced_2_45.png'),
(31, 2, 46, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.28030460405089996, \"match_val\": 0.5956995487213135, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_2_46.png', 'static/roi_images\\roi_produced_2_46.png'),
(32, 2, 47, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.2779431720581496, \"match_val\": 0.5866715312004089, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_2_47.png', 'static/roi_images\\roi_produced_2_47.png'),
(33, 2, 48, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.3401638370842099, \"match_val\": 0.6081954836845398, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_2_48.png', 'static/roi_images\\roi_produced_2_48.png'),
(34, 2, 49, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.263715651556261, \"match_val\": 0.5744544267654419, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_2_49.png', 'static/roi_images\\roi_produced_2_49.png'),
(35, 2, 50, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.21938964585493184, \"match_val\": 0.5468843579292297, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_2_50.png', 'static/roi_images\\roi_produced_2_50.png'),
(36, 2, 55, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.32299095808227063, \"match_val\": 0.5133858323097229, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_2_55.png', 'static/roi_images\\roi_produced_2_55.png'),
(37, 2, 56, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.2961283614006516, \"match_val\": 0.4848628044128418, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_2_56.png', 'static/roi_images\\roi_produced_2_56.png'),
(38, 2, 57, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.2870785488339129, \"match_val\": 0.501654326915741, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_2_57.png', 'static/roi_images\\roi_produced_2_57.png'),
(39, 3, 51, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.3315637733843529, \"match_val\": 0.6677373647689819, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_3_51_97b744f9-4341-4e32-a40d-2f870abd42ab.png', 'static/roi_images\\roi_produced_3_51_fa4567c8-1234-4912-a884-4aabf621161f.png'),
(40, 3, 52, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.22299658302198874, \"match_val\": 0.6567292213439941, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_3_52_e5e35eff-23f5-4929-8457-95b843bf8c50.png', 'static/roi_images\\roi_produced_3_52_feec9d18-0c52-427d-93c4-c3ad6a13d125.png'),
(41, 3, 53, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.2863121400799731, \"match_val\": 0.6490955948829651, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_3_53_54ab9ddc-c26b-4162-bf67-5142d8f28419.png', 'static/roi_images\\roi_produced_3_53_488464b2-a98b-4fd3-8b4b-6b8cce566364.png'),
(42, 3, 54, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.28049549953773784, \"match_val\": 0.550318717956543, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_3_54_531ac668-fee3-4fce-94f7-5af7bb63be86.png', 'static/roi_images\\roi_produced_3_54_68743a56-0bee-4998-a824-64813dd3702c.png'),
(43, 3, 58, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.42385472651103545, \"match_val\": 0.7932224869728088, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_3_58_78997175-09d3-4a29-ba90-69fb1b32a04b.png', 'static/roi_images\\roi_produced_3_58_203c901a-1add-492f-b691-a81801323d1d.png'),
(44, 3, 59, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.3619640233608222, \"match_val\": 0.7679339647293091, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_3_59_9317678f-c03f-4bf2-b610-df6e535a8f1b.png', 'static/roi_images\\roi_produced_3_59_5ddcab47-26b0-4cde-9ca2-c23504f23a58.png'),
(45, 3, 60, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.45782079341954857, \"match_val\": 0.7747188806533813, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_3_60_158d1498-d99f-4aed-b348-8b01ba596f6e.png', 'static/roi_images\\roi_produced_3_60_7a6bff2b-d59d-4d83-90ad-ee0a186ded7f.png'),
(46, 3, 42, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.23821836828964216, \"match_val\": 0.5279262065887451, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_3_42_280a2d97-a9b7-43ee-af85-bacca07938f4.png', 'static/roi_images\\roi_produced_3_42_d36b722e-8f39-491e-b879-09f35846f138.png'),
(47, 3, 43, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.25628743457436876, \"match_val\": 0.5374245643615723, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_3_43_29b84c8d-6842-4806-81e1-b4e7397c64f1.png', 'static/roi_images\\roi_produced_3_43_ea4d21c4-c377-4638-a51e-d2092751c429.png'),
(48, 3, 44, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.3475131016379556, \"match_val\": 0.5773675441741943, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_3_44_d53f7acf-f423-4d71-92ec-fffa0f81c1bf.png', 'static/roi_images\\roi_produced_3_44_c0f20624-eda0-4426-a356-0a8d4d17288e.png'),
(49, 3, 45, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.25543370514694363, \"match_val\": 0.5323603749275208, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_3_45_3fd428c3-3ab4-43e6-850f-93935fcb7e64.png', 'static/roi_images\\roi_produced_3_45_72ffdd2d-c7a7-4a80-a4cf-bd2681061a78.png'),
(50, 3, 46, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.28030460405089996, \"match_val\": 0.5398672223091125, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_3_46_192bebc5-b779-42e6-b410-425fa7887763.png', 'static/roi_images\\roi_produced_3_46_fc9a1b74-b4f8-42a1-b6d3-a1aca5483fb2.png'),
(51, 3, 47, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.2779431720581496, \"match_val\": 0.5287326574325562, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_3_47_f013fb83-25fb-4812-a2dc-3afdfa26c1d1.png', 'static/roi_images\\roi_produced_3_47_a9267138-ccde-4b3e-9456-4ac3c33437c9.png'),
(52, 3, 48, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.3401638370842099, \"match_val\": 0.552306592464447, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_3_48_d72e3cbc-1d1b-4f0a-a882-114a2aa62c25.png', 'static/roi_images\\roi_produced_3_48_08bc375a-3d9e-4b0b-9c9e-6b3291f1609e.png'),
(53, 3, 49, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.263715651556261, \"match_val\": 0.5130578279495239, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_3_49_b7a342d0-69d5-4f7e-bf10-c819ec8fffc1.png', 'static/roi_images\\roi_produced_3_49_289a4f2b-cc18-443c-8f90-ca99b7496218.png'),
(54, 3, 50, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.21938964585493184, \"match_val\": 0.48401251435279846, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_3_50_d549b6c6-c547-49dd-8894-135c5157a2d2.png', 'static/roi_images\\roi_produced_3_50_c44fa7de-8884-45b6-a730-a24b432d19df.png'),
(55, 3, 55, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.32299095808227063, \"match_val\": 0.44973745942115784, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_3_55_09efc1d9-c419-4ae1-9cb6-e3bfe200d4d7.png', 'static/roi_images\\roi_produced_3_55_7ad83350-760b-4919-9c4f-96001e53b403.png'),
(56, 3, 56, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.2961283614006516, \"match_val\": 0.4179430603981018, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_3_56_d7494d05-535c-42bb-ad54-905c98e4bad2.png', 'static/roi_images\\roi_produced_3_56_c0b6170b-4459-4b0a-9e10-a710d4d0d5bc.png'),
(57, 3, 57, 'FAIL', '0°', 0, 0, '{\"ssim\": 0.2870785488339129, \"match_val\": 0.43690767884254456, \"mode\": \"Masked Template\"}', 'static/roi_images\\roi_golden_3_57_a3e2f1af-08cf-420e-adec-15d234b47661.png', 'static/roi_images\\roi_produced_3_57_fedd88be-37d8-47be-8880-905b7f291f29.png');

-- --------------------------------------------------------

--
-- Table structure for table `inspection_results_old`
--
-- Error reading structure for table smt_inspection_new.inspection_results_old: #1932 - Table 'smt_inspection_new.inspection_results_old' doesn't exist in engine
-- Error reading data for table smt_inspection_new.inspection_results_old: #1064 - You have an error in your SQL syntax; check the manual that corresponds to your MariaDB server version for the right syntax to use near 'FROM `smt_inspection_new`.`inspection_results_old`' at line 1

-- --------------------------------------------------------

--
-- Table structure for table `packages`
--

CREATE TABLE `packages` (
  `id` int(11) NOT NULL,
  `name` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `packages`
--

INSERT INTO `packages` (`id`, `name`) VALUES
(25, '0204'),
(54, 'C1'),
(28, 'CAP01'),
(1, 'generic'),
(61, 'ODD1'),
(40, 'PAC1'),
(38, 'R1'),
(27, 'SOT23');

-- --------------------------------------------------------

--
-- Table structure for table `products`
--

CREATE TABLE `products` (
  `id` int(11) NOT NULL,
  `name` varchar(255) NOT NULL,
  `golden_image` varchar(255) NOT NULL,
  `fiducials` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL CHECK (json_valid(`fiducials`))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `products`
--

INSERT INTO `products` (`id`, `name`, `golden_image`, `fiducials`) VALUES
(1, 'Novo1', 'static/uploads\\golden_4fiduciais.png', '[{\"x\":77,\"y\":48,\"r\":22,\"name\":\"FID1\"},{\"x\":73,\"y\":516,\"r\":22,\"name\":\"FID2\"},{\"x\":928,\"y\":48,\"r\":21,\"name\":\"FID3\"},{\"x\":924,\"y\":516,\"r\":22,\"name\":\"FID4\"}]'),
(2, 'Bananinha', 'static/uploads\\golden_4fiduciais.png', '[{\"x\":77,\"y\":48,\"r\":22,\"name\":\"FID1\"},{\"x\":928,\"y\":48,\"r\":21,\"name\":\"FID2\"},{\"x\":73,\"y\":516,\"r\":22,\"name\":\"FID3\"},{\"x\":924,\"y\":516,\"r\":22,\"name\":\"FID4\"}]'),
(3, 'Placa 01', 'static/uploads\\golden_4fiduciais.png', '[{\"x\":77,\"y\":48,\"r\":22,\"name\":\"FID1\"},{\"x\":73,\"y\":516,\"r\":22,\"name\":\"FID2\"},{\"x\":928,\"y\":48,\"r\":21,\"name\":\"FID3\"},{\"x\":924,\"y\":516,\"r\":22,\"name\":\"FID4\"}]'),
(4, 'Placa Completa', 'static/uploads\\golden_4fiduciais.png', '[{\"x\":77,\"y\":48,\"r\":22,\"name\":\"FID1\"},{\"x\":73,\"y\":516,\"r\":22,\"name\":\"FID2\"},{\"x\":928,\"y\":48,\"r\":21,\"name\":\"FID3\"},{\"x\":924,\"y\":516,\"r\":22,\"name\":\"FID4\"}]'),
(5, 'teste novo', 'static/uploads\\golden_4fiduciais.png', '[{\"x\":77,\"y\":48,\"r\":22,\"name\":\"FID1\"},{\"x\":73,\"y\":516,\"r\":22,\"name\":\"FID2\"},{\"x\":928,\"y\":48,\"r\":21,\"name\":\"FID3\"},{\"x\":924,\"y\":516,\"r\":22,\"name\":\"FID4\"}]'),
(6, '001', 'static/uploads\\golden_4fiduciais.png', '[{\"x\":77,\"y\":48,\"r\":22,\"name\":\"FID1\"},{\"x\":73,\"y\":516,\"r\":22,\"name\":\"FID2\"},{\"x\":924,\"y\":516,\"r\":22,\"name\":\"FID3\"},{\"x\":928,\"y\":48,\"r\":21,\"name\":\"FID4\"}]'),
(7, '002rotacao', 'static/uploads\\golden_4fiduciais.png', '[{\"x\":77,\"y\":48,\"r\":22,\"name\":\"FID1\"},{\"x\":73,\"y\":516,\"r\":22,\"name\":\"FID2\"},{\"x\":928,\"y\":48,\"r\":21,\"name\":\"FID3\"},{\"x\":924,\"y\":516,\"r\":22,\"name\":\"FID4\"}]'),
(8, '003', 'static/uploads\\golden_4fiduciais.png', '[{\"x\":77,\"y\":48,\"r\":22,\"name\":\"FID1\"},{\"x\":73,\"y\":516,\"r\":22,\"name\":\"FID2\"},{\"x\":928,\"y\":48,\"r\":21,\"name\":\"FID3\"},{\"x\":924,\"y\":516,\"r\":22,\"name\":\"FID4\"}]'),
(9, 'w41', 'static/uploads\\golden_4fiduciais.png', '[{\"name\":\"FID1\",\"x\":77,\"y\":48,\"r\":22},{\"name\":\"FID2\",\"x\":73,\"y\":516,\"r\":22},{\"name\":\"FID3\",\"x\":928,\"y\":48,\"r\":21},{\"name\":\"FID4\",\"x\":924,\"y\":516,\"r\":22}]'),
(10, 'w41-0', 'static/uploads\\golden_4fiduciais.png', '[{\"name\":\"FID1\",\"x\":77,\"y\":48,\"r\":22},{\"name\":\"FID2\",\"x\":73,\"y\":516,\"r\":22},{\"name\":\"FID3\",\"x\":924,\"y\":516,\"r\":22},{\"name\":\"FID4\",\"x\":928,\"y\":48,\"r\":21}]'),
(11, 'w41-0', 'static/uploads\\golden_4fiduciais.png', '[{\"name\":\"FID1\",\"x\":77,\"y\":48,\"r\":22},{\"name\":\"FID2\",\"x\":73,\"y\":516,\"r\":22},{\"name\":\"FID3\",\"x\":924,\"y\":516,\"r\":22},{\"name\":\"FID4\",\"x\":928,\"y\":48,\"r\":21}]'),
(12, 'w41-0', 'static/uploads\\golden_4fiduciais.png', '[{\"name\":\"FID1\",\"x\":77,\"y\":48,\"r\":22},{\"name\":\"FID2\",\"x\":73,\"y\":516,\"r\":22},{\"name\":\"FID3\",\"x\":928,\"y\":48,\"r\":21},{\"name\":\"FID4\",\"x\":924,\"y\":516,\"r\":22}]'),
(13, 'w41-0', 'static/uploads\\golden_4fiduciais.png', '[{\"name\":\"FID1\",\"x\":77,\"y\":48,\"r\":22},{\"name\":\"FID2\",\"x\":73,\"y\":516,\"r\":22},{\"name\":\"FID3\",\"x\":924,\"y\":516,\"r\":22},{\"name\":\"FID4\",\"x\":928,\"y\":48,\"r\":21}]'),
(14, 'w41-new', 'static/uploads\\golden_4fiduciais.png', '[{\"name\":\"FID1\",\"x\":77,\"y\":48,\"r\":22},{\"name\":\"FID2\",\"x\":73,\"y\":516,\"r\":22},{\"name\":\"FID3\",\"x\":928,\"y\":48,\"r\":21},{\"name\":\"FID4\",\"x\":924,\"y\":516,\"r\":22}]');

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL,
  `username` varchar(255) NOT NULL,
  `password_hash` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`id`, `username`, `password_hash`) VALUES
(1, 'admin', 'scrypt:32768:8:1$4cmmhlUVFXTSWYyZ$4339f4bb37f5745a761a175d326fa3155b495ae29019303a9caa0ff186474597ef7d84b6e6deb6d33bb72b2b0c663a61315901a2c942761b6afd20fde7132826');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `components`
--
ALTER TABLE `components`
  ADD PRIMARY KEY (`id`),
  ADD KEY `product_id` (`product_id`),
  ADD KEY `fk_components_package` (`package_id`);

--
-- Indexes for table `inspections`
--
ALTER TABLE `inspections`
  ADD PRIMARY KEY (`id`),
  ADD KEY `product_id` (`product_id`);

--
-- Indexes for table `inspection_results`
--
ALTER TABLE `inspection_results`
  ADD PRIMARY KEY (`id`),
  ADD KEY `inspection_id` (`inspection_id`);

--
-- Indexes for table `packages`
--
ALTER TABLE `packages`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `name` (`name`);

--
-- Indexes for table `products`
--
ALTER TABLE `products`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `username` (`username`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `components`
--
ALTER TABLE `components`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=61;

--
-- AUTO_INCREMENT for table `inspections`
--
ALTER TABLE `inspections`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT for table `inspection_results`
--
ALTER TABLE `inspection_results`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=58;

--
-- AUTO_INCREMENT for table `packages`
--
ALTER TABLE `packages`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=64;

--
-- AUTO_INCREMENT for table `products`
--
ALTER TABLE `products`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=15;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `components`
--
ALTER TABLE `components`
  ADD CONSTRAINT `components_ibfk_1` FOREIGN KEY (`product_id`) REFERENCES `products` (`id`) ON DELETE CASCADE,
  ADD CONSTRAINT `components_ibfk_2` FOREIGN KEY (`package_id`) REFERENCES `packages` (`id`) ON DELETE SET NULL,
  ADD CONSTRAINT `fk_components_package` FOREIGN KEY (`package_id`) REFERENCES `packages` (`id`);

--
-- Constraints for table `inspections`
--
ALTER TABLE `inspections`
  ADD CONSTRAINT `inspections_ibfk_1` FOREIGN KEY (`product_id`) REFERENCES `products` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `inspection_results`
--
ALTER TABLE `inspection_results`
  ADD CONSTRAINT `inspection_results_ibfk_1` FOREIGN KEY (`inspection_id`) REFERENCES `inspections` (`id`) ON DELETE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
