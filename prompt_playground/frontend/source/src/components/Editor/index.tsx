import React from "react";
import { Button, Col, Container, Row } from "react-bootstrap";

import { useState } from 'react';
import ParamsEditor from "./ParamsEditor";
import TextEditor from "./TextEditor";

type TextEditorProps = {
    text: string;
}

type ParamsEditorProps = {
    mode: string;
    model: string;
    temperature: number;
    maxLength: number;

}

const Editor: React.FC = () => {
    const [state, setState] = useState<TextEditorProps>({
        text: '',
    })
    // console.log(state);

    const handleSubmit = ((e: React.MouseEvent<HTMLButtonElement>) => {
        console.log(e);
        console.log(state);
    });

    const handleChange = ((e: React.ChangeEvent<HTMLInputElement>) => {
        setState({
            text: e.target.value,
        })
    });

    return (
        <Container>
            <Row>
                <Col xs={9}>
                    <TextEditor
                        onChange={(e) => { handleChange(e) }} />
                    <Button variant="outline-primary" type="submit" id="submit-prompt" onClick={handleSubmit}>
                        Submit
                    </Button>
                </Col>
                <Col>
                    <ParamsEditor />
                </Col>
            </Row>
        </Container>
    )
}

export default Editor;