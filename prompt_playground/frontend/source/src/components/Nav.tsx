import React from "react";
import {
    Container,
    Navbar
} from "react-bootstrap";

import logo from '../imgs/logo192.png';

const Nav: React.FC = () => {
    return (
        <Navbar bg="dark" variant="dark" className="mb-3">
            <Container>
                <Navbar.Brand href="#">
                    <img
                        alt=""
                        src={logo}
                        width="30"
                        height="30"
                        className="d-inline-block align-top"
                    />{' '}
                    Prompt Playground
                </Navbar.Brand>
            </Container>
        </Navbar>
    )
}

export default Nav;